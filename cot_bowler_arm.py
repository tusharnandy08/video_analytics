"""
Chain-of-thought probe.

Hypothesis: in the previous runs, Pegasus 1.5 produced an identical
templated `evidence` string for all 10 deliveries and flipped its answer
when the prompt was reframed. The question we want to answer here is
whether forcing the model to *describe* what it sees before *classifying*
the arm changes the outcome — i.e. whether CoT breaks the pattern, or
whether the model is genuinely blind to bowling-arm pixels.

Design:

1. Field order in the JSON schema is "describe first, classify last".
   LLMs fill JSON fields in declaration order, so this is the actual
   mechanism that forces a CoT-shaped output.

2. The prompt does NOT mention "left" or "right" anywhere. The previous
   runs flipped on the prompt's framing word; if we don't supply a word,
   the model has to source the answer from its own description.

3. Each delivery carries a free-text `visible_action` and
   `arm_at_release_observation`. After the run we check whether those
   strings are unique per delivery (real observations) or identical
   across deliveries (templating, same failure mode as before).

4. Per-delivery `confidence` is asked for — does the model self-flag
   uncertainty when it can't actually see the release?

Reuses the asset_id cached in .pipeline_state.json.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from dotenv import dotenv_values
from twelvelabs import TwelveLabs
from twelvelabs.types import AsyncResponseFormat, VideoContext_AssetId

ROOT = Path(__file__).parent

PROMPT = (
    "This video is a cricket highlights compilation. For every distinct "
    "bowling delivery shown, work through your observations in this exact "
    "order, and only commit to a classification at the end:\n"
    "\n"
    "1. visible_action — describe in detail what you can actually see "
    "during the delivery: the bowler's run-up, body posture, the angle of "
    "the camera, the visibility of the release, and any distinguishing "
    "features (kit colour, build, stadium, etc.). Be specific about what "
    "is visible vs. obscured. Do not speculate.\n"
    "\n"
    "2. arm_at_release_observation — describe specifically what you "
    "observe at the instant the ball leaves the bowler's hand. Which arm "
    "is raised above the head? Which arm is at the side? Which hand is "
    "the ball released from? If the camera angle does not clearly show "
    "this, say so explicitly.\n"
    "\n"
    "3. bowling_arm — only after writing the two descriptions above, "
    "classify the bowler's bowling arm. Use 'unknown' whenever your "
    "arm_at_release_observation does not establish the arm with "
    "confidence — do not guess.\n"
    "\n"
    "4. confidence — self-rate how sure you are about the bowling_arm "
    "classification given what you actually saw.\n"
    "\n"
    "Provide approximate start/end timestamps for each delivery. "
    "Aggregate the counts at the end."
)

# Field order is load-bearing. The descriptive fields MUST come before
# bowling_arm so the model writes its observations before committing to
# a label. If we reordered this with bowling_arm first, the model would
# emit the label first and then justify it — the opposite of CoT.
SCHEMA = {
    "type": "object",
    "properties": {
        "deliveries": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "delivery_number": {"type": "integer"},
                    "approximate_start_seconds": {"type": "number"},
                    "approximate_end_seconds": {"type": "number"},
                    "visible_action": {"type": "string"},
                    "arm_at_release_observation": {"type": "string"},
                    "bowling_arm": {
                        "type": "string",
                        "enum": ["left", "right", "unknown"],
                    },
                    "confidence": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                    },
                },
                "required": [
                    "delivery_number",
                    "approximate_start_seconds",
                    "approximate_end_seconds",
                    "visible_action",
                    "arm_at_release_observation",
                    "bowling_arm",
                    "confidence",
                ],
            },
        },
        "left_arm_count": {"type": "integer"},
        "right_arm_count": {"type": "integer"},
        "unknown_count": {"type": "integer"},
        "total_deliveries": {"type": "integer"},
        "reasoning_summary": {"type": "string"},
    },
    "required": [
        "deliveries",
        "left_arm_count",
        "right_arm_count",
        "unknown_count",
        "total_deliveries",
        "reasoning_summary",
    ],
}


def main() -> None:
    env = dotenv_values(ROOT / ".env")
    api_key = env.get("twelvelabs-api-key") or env.get("TWELVELABS_API_KEY")
    state = json.loads((ROOT / ".pipeline_state.json").read_text())
    asset_id = state["asset_id"]

    client = TwelveLabs(api_key=api_key)
    print(f"[cot] asset_id={asset_id}, model=pegasus1.5, temperature=0")
    task = client.analyze_async.tasks.create(
        video=VideoContext_AssetId(asset_id=asset_id),
        model_name="pegasus1.5",
        prompt=PROMPT,
        temperature=0.0,
        response_format=AsyncResponseFormat(type="json_schema", json_schema=SCHEMA),
    )
    print(f"[cot] task_id={task.task_id}")

    while True:
        task = client.analyze_async.tasks.retrieve(task.task_id)
        if task.status == "ready":
            break
        if task.status == "failed":
            raise RuntimeError(f"Failed: {task.error}")
        print(f"[cot] status={task.status} ...")
        time.sleep(10)

    result = json.loads(task.result.data)
    (ROOT / "last_cot_result.json").write_text(json.dumps(result, indent=2))

    deliveries = result["deliveries"]

    # The diagnostic that matters: are the descriptions actually distinct
    # per delivery, or templated?
    visible = [d["visible_action"] for d in deliveries]
    release = [d["arm_at_release_observation"] for d in deliveries]
    n_unique_visible = len(set(visible))
    n_unique_release = len(set(release))

    print()
    print("=" * 72)
    print("CHAIN-OF-THOUGHT RESULT")
    print("=" * 72)
    print(f"Total deliveries  : {result['total_deliveries']}")
    print(f"Left arm          : {result['left_arm_count']}")
    print(f"Right arm         : {result['right_arm_count']}")
    print(f"Unknown           : {result['unknown_count']}")
    print()
    print(f"Distinct visible_action strings              : {n_unique_visible}/{len(visible)}")
    print(f"Distinct arm_at_release_observation strings  : {n_unique_release}/{len(release)}")
    print()
    print(f"Summary: {result['reasoning_summary']}")
    print()
    for d in deliveries:
        print(
            f"#{d['delivery_number']:>2}  "
            f"{d['approximate_start_seconds']:6.1f}s – {d['approximate_end_seconds']:6.1f}s  "
            f"arm={d['bowling_arm']:<7}  conf={d['confidence']}"
        )
        print(f"     visible:  {d['visible_action']}")
        print(f"     release:  {d['arm_at_release_observation']}")
    print("=" * 72)


if __name__ == "__main__":
    main()

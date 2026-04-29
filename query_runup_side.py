"""
Probe: ask Pegasus 1.5 the inverse / sibling question.

Pegasus 1.5 just told us every bowler ran in "from the left side of the
screen" with an identical evidence string for all 10. If that's a real
observation, asking "how many bowled from the RIGHT side of the screen"
should return 0. If it's templated boilerplate, the model may flip its
answer when the question is reframed.

Reuses the asset_id from .pipeline_state.json.
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
    "For every distinct bowling delivery in this cricket highlights video, "
    "look at the camera shot of the bowler's run-up and release. "
    "Identify which side of the screen the bowler runs in from — 'left' if "
    "the bowler enters frame from the left and runs toward the right, "
    "'right' if the bowler enters frame from the right and runs toward the "
    "left, 'unknown' if the angle does not show the run-up. "
    "Provide approximate start/end timestamps and short evidence. "
    "Then count how many deliveries had a right-side run-up."
)

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
                    "run_up_side": {
                        "type": "string",
                        "enum": ["left", "right", "unknown"],
                    },
                    "evidence": {"type": "string"},
                },
                "required": [
                    "delivery_number",
                    "approximate_start_seconds",
                    "approximate_end_seconds",
                    "run_up_side",
                    "evidence",
                ],
            },
        },
        "left_side_runup_count": {"type": "integer"},
        "right_side_runup_count": {"type": "integer"},
        "unknown_count": {"type": "integer"},
        "total_deliveries": {"type": "integer"},
        "summary": {"type": "string"},
    },
    "required": [
        "deliveries",
        "left_side_runup_count",
        "right_side_runup_count",
        "unknown_count",
        "total_deliveries",
        "summary",
    ],
}


def main() -> None:
    env = dotenv_values(ROOT / ".env")
    api_key = env.get("twelvelabs-api-key") or env.get("TWELVELABS_API_KEY")
    state = json.loads((ROOT / ".pipeline_state.json").read_text())
    asset_id = state["asset_id"]

    client = TwelveLabs(api_key=api_key)
    print(f"[query] asset_id={asset_id}, model=pegasus1.5")
    task = client.analyze_async.tasks.create(
        video=VideoContext_AssetId(asset_id=asset_id),
        model_name="pegasus1.5",
        prompt=PROMPT,
        response_format=AsyncResponseFormat(type="json_schema", json_schema=SCHEMA),
    )
    print(f"[query] task_id={task.task_id}")

    while True:
        task = client.analyze_async.tasks.retrieve(task.task_id)
        if task.status == "ready":
            break
        if task.status == "failed":
            raise RuntimeError(f"Failed: {task.error}")
        print(f"[query] status={task.status} ...")
        time.sleep(10)

    result = json.loads(task.result.data)
    (ROOT / "last_runup_result.json").write_text(json.dumps(result, indent=2))

    print()
    print("=" * 70)
    print("RUN-UP SIDE")
    print("=" * 70)
    print(f"Total deliveries  : {result['total_deliveries']}")
    print(f"Left-side run-up  : {result['left_side_runup_count']}")
    print(f"Right-side run-up : {result['right_side_runup_count']}")
    print(f"Unknown           : {result['unknown_count']}")
    print()
    print(f"Summary: {result['summary']}")
    print()
    for d in result["deliveries"]:
        ts = f"{d['approximate_start_seconds']:6.1f}s – {d['approximate_end_seconds']:6.1f}s"
        print(f"  #{d['delivery_number']:>2}  {ts}  side={d['run_up_side']:<7}  {d['evidence']}")
    print("=" * 70)


if __name__ == "__main__":
    main()

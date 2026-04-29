"""
Pipeline: count left-arm bowlers in a cricket video using the Twelve Labs API.

Stages:
  1. Load API key from .env
  2. Create (or reuse) an index with Marengo 3.0 + Pegasus 1.2
  3. Upload the local mp4 as an asset
  4. Index the asset and poll until ready
  5. Call /analyze with a JSON schema that asks Pegasus to enumerate every
     bowling delivery and classify the bowler's bowling arm
  6. Count and print the result

Run state is cached in .pipeline_state.json so re-runs skip work that has
already succeeded.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from dotenv import dotenv_values
from twelvelabs import TwelveLabs
from twelvelabs.types import (
    AsyncResponseFormat,
    SyncResponseFormat,
    VideoContext_AssetId,
)

ROOT = Path(__file__).parent
VIDEO_PATH = ROOT / ".videos" / "sachin_straight_drives.mp4"
STATE_PATH = ROOT / ".pipeline_state.json"
INDEX_NAME = "cricket-handedness-experiment"

# Pegasus 1.5 is async-only and works directly on an asset (no indexing).
# Pegasus 1.2 is sync and requires the asset to be indexed first.
PEGASUS_MODEL = os.environ.get("PEGASUS_MODEL", "pegasus1.5")

# Schema we hand to Pegasus. Asking for evidence per delivery makes the
# answer auditable — we can sanity-check against the timestamps ourselves.
ANALYZE_SCHEMA = {
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
                    "bowler_arm": {
                        "type": "string",
                        "enum": ["left", "right", "unknown"],
                    },
                    "evidence": {"type": "string"},
                },
                "required": [
                    "delivery_number",
                    "approximate_start_seconds",
                    "approximate_end_seconds",
                    "bowler_arm",
                    "evidence",
                ],
            },
        },
        "left_arm_bowler_count": {"type": "integer"},
        "right_arm_bowler_count": {"type": "integer"},
        "unknown_count": {"type": "integer"},
        "total_deliveries": {"type": "integer"},
        "summary": {"type": "string"},
    },
    "required": [
        "deliveries",
        "left_arm_bowler_count",
        "right_arm_bowler_count",
        "unknown_count",
        "total_deliveries",
        "summary",
    ],
}

ANALYZE_PROMPT = (
    "This is a highlights compilation of cricket batsman Sachin Tendulkar "
    "playing straight drives. For every distinct bowling delivery shown in "
    "the video, identify the bowler's bowling arm — i.e. which arm they use "
    "to release the ball, NOT the batsman's handedness. "
    "Use 'left' for left-arm bowlers, 'right' for right-arm bowlers, and "
    "'unknown' only when the run-up or release is not visible. "
    "For each delivery, provide approximate start/end timestamps in seconds, "
    "and a short evidence string describing what was visible (run-up "
    "direction, release arm, follow-through). "
    "Then total the counts. The answer to 'how many bowlers were left-handed' "
    "is left_arm_bowler_count."
)


def load_api_key() -> str:
    env = dotenv_values(ROOT / ".env")
    for k in ("twelvelabs-api-key", "TWELVELABS_API_KEY", "TWELVE_LABS_API_KEY"):
        if env.get(k):
            return env[k]
    raise RuntimeError("No Twelve Labs API key found in .env")


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {}


def save_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2))


def get_or_create_index(client: TwelveLabs, state: dict) -> str:
    if state.get("index_id"):
        print(f"[index]   reusing {state['index_id']}")
        return state["index_id"]

    print(f"[index]   creating '{INDEX_NAME}' (marengo3.0 + pegasus1.2)")
    index = client.indexes.create(
        index_name=INDEX_NAME,
        models=[
            {"model_name": "marengo3.0", "model_options": ["visual", "audio"]},
            {"model_name": "pegasus1.2", "model_options": ["visual", "audio"]},
        ],
    )
    state["index_id"] = index.id
    save_state(state)
    print(f"[index]   created  {index.id}")
    return index.id


def get_or_create_asset(client: TwelveLabs, state: dict) -> str:
    if state.get("asset_id"):
        print(f"[asset]   reusing {state['asset_id']}")
        return state["asset_id"]

    size_mb = VIDEO_PATH.stat().st_size / (1024 * 1024)
    print(f"[asset]   uploading {VIDEO_PATH.name} ({size_mb:.1f} MB)")
    with open(VIDEO_PATH, "rb") as fh:
        asset = client.assets.create(method="direct", file=fh)

    while True:
        asset = client.assets.retrieve(asset.id)
        if asset.status == "ready":
            break
        if asset.status == "failed":
            raise RuntimeError(f"Asset upload failed: {asset.id}")
        print(f"[asset]   status={asset.status} ...")
        time.sleep(5)

    state["asset_id"] = asset.id
    save_state(state)
    print(f"[asset]   ready   {asset.id}")
    return asset.id


def index_asset(client: TwelveLabs, state: dict, index_id: str, asset_id: str) -> str:
    if state.get("indexed_asset_id"):
        print(f"[indexed] reusing {state['indexed_asset_id']}")
        return state["indexed_asset_id"]

    print(f"[indexed] indexing asset into {index_id}")
    indexed = client.indexes.indexed_assets.create(
        index_id=index_id, asset_id=asset_id
    )

    while True:
        indexed = client.indexes.indexed_assets.retrieve(
            index_id=index_id, indexed_asset_id=indexed.id
        )
        if indexed.status == "ready":
            break
        if indexed.status == "failed":
            raise RuntimeError(f"Indexing failed: {indexed.id}")
        print(f"[indexed] status={indexed.status} ...")
        time.sleep(10)

    state["indexed_asset_id"] = indexed.id
    save_state(state)
    print(f"[indexed] ready   {indexed.id}")
    return indexed.id


def analyze_sync(client: TwelveLabs, asset_id: str) -> dict:
    print("[analyze] calling Pegasus 1.2 (sync) with structured schema")
    response = client.analyze(
        video=VideoContext_AssetId(asset_id=asset_id),
        prompt=ANALYZE_PROMPT,
        response_format=SyncResponseFormat(
            type="json_schema",
            json_schema=ANALYZE_SCHEMA,
        ),
    )
    if not response.data:
        raise RuntimeError(f"Empty response (finish_reason={response.finish_reason})")
    return json.loads(response.data)


def analyze_async(client: TwelveLabs, asset_id: str, model_name: str) -> dict:
    print(f"[analyze] submitting async task with {model_name}")
    task = client.analyze_async.tasks.create(
        video=VideoContext_AssetId(asset_id=asset_id),
        model_name=model_name,
        prompt=ANALYZE_PROMPT,
        response_format=AsyncResponseFormat(
            type="json_schema",
            json_schema=ANALYZE_SCHEMA,
        ),
    )
    print(f"[analyze] task_id={task.task_id}")

    while True:
        task = client.analyze_async.tasks.retrieve(task.task_id)
        if task.status == "ready":
            break
        if task.status == "failed":
            err = getattr(task, "error", None)
            raise RuntimeError(f"Analyze task failed: {err}")
        print(f"[analyze] status={task.status} ...")
        time.sleep(10)

    if not task.result or not task.result.data:
        raise RuntimeError(f"Empty result (finish_reason={getattr(task.result,'finish_reason',None)})")
    return json.loads(task.result.data)


def analyze(client: TwelveLabs, asset_id: str) -> dict:
    if PEGASUS_MODEL == "pegasus1.2":
        return analyze_sync(client, asset_id)
    return analyze_async(client, asset_id, PEGASUS_MODEL)


def render_report(result: dict) -> None:
    print()
    print("=" * 70)
    print(f"RESULT  (model={PEGASUS_MODEL})")
    print("=" * 70)
    print(f"Total deliveries identified : {result['total_deliveries']}")
    print(f"Left-arm bowlers            : {result['left_arm_bowler_count']}")
    print(f"Right-arm bowlers           : {result['right_arm_bowler_count']}")
    print(f"Unknown                     : {result['unknown_count']}")
    print()
    print("Summary:")
    print(f"  {result['summary']}")
    print()
    print("Per-delivery breakdown:")
    for d in result["deliveries"]:
        ts = f"{d['approximate_start_seconds']:6.1f}s – {d['approximate_end_seconds']:6.1f}s"
        print(f"  #{d['delivery_number']:>2}  {ts}  arm={d['bowler_arm']:<7}  {d['evidence']}")
    print("=" * 70)


def main() -> int:
    if not VIDEO_PATH.exists():
        print(f"Video not found at {VIDEO_PATH}", file=sys.stderr)
        return 1

    client = TwelveLabs(api_key=load_api_key())
    state = load_state()

    index_id = get_or_create_index(client, state)
    asset_id = get_or_create_asset(client, state)
    index_asset(client, state, index_id, asset_id)

    result = analyze(client, asset_id)
    (ROOT / "last_result.json").write_text(json.dumps(result, indent=2))
    render_report(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())

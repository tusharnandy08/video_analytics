# video_analytics

A scratchpad for stress-testing the current generation of video-understanding
tools. Each experiment lives in its own pipeline and answers one narrow
question about a real video, so we can build a feel for where these models
shine and where they fall over.

## Experiment 1 — Twelve Labs (Pegasus 1.2 → Pegasus 1.5)

### What we read

The [Pegasus 1.5 launch post](https://www.twelvelabs.io/blog/introducing-pegasus-1-5).
The headline pitch is a *"fundamental shift in video understanding: from
answering questions about clips to generating structured, time-based
metadata across entire videos"*. The conclusion goes further: *"video
stops being something you watch and becomes something your systems can
compute over."* Verification, in the post, is defined as **JSON structural
validity** — schema conformance, valid enums. The discussed failure modes
are all about segment **boundaries** (over-segmentation, single-segment
collapse, boundary ambiguity). The evaluation set is news, movies/TV, and
sports (basketball). No benchmark numbers, no embedded video, no
1.2-vs-1.5 comparison example.

### What we saw

The 31-minute [deep-dive video](https://www.youtube.com/watch?v=7QBO1abf7eQ)
with CTO Aiden, Head of Product Travis Couture, and Pegasus Research Lead
Kian Kim. Three live demos: news-broadcast speaker segmentation, a
"needle in a haystack" visual object detection demo, and sports content
with multiple segment definitions. The chapter at
[5:04 — Evaluation Framework and Reinforcement Learning](https://www.youtube.com/watch?v=7QBO1abf7eQ&t=301s)
is the load-bearing one for our experiment: they explain that Pegasus 1.5
was trained with **RLVR** (reinforcement learning with verifiable
rewards), and the rewards being optimised are **segmentation F1** plus
**format validity**. That is, the training signal scores *where* the
segments fall and *whether the JSON is well-formed* — not *whether the
metadata inside each segment is grounded in the frames*.

### The experiment we ran

Source video: [Sachin Tendulkar's Top 10 Straight Drives](https://www.youtube.com/watch?v=oruAq_GSLZ4)
(234 s, fast-cut compilation with slow-mo replays).

**Question.** *How many of the bowlers shown are left-arm bowlers?*

A clean stress-test of fine-grained visual reasoning: the model has to
find each delivery, watch the run-up / release / follow-through (often a
fraction of a second of clean view), and decide which arm released the
ball — not which side the batsman is on (Sachin is right-handed, which is
the easy distractor).

The pipeline ([count_left_handed_bowlers.py](count_left_handed_bowlers.py)):

1. Fetch the source as mp4 via `yt-dlp` (no transcoding).
2. Create a Twelve Labs index with `marengo3.0` + `pegasus1.2`.
3. Upload as an asset, then index the asset.
4. Call `client.analyze(...)` (Pegasus 1.2, sync) **or** submit
   `client.analyze_async.tasks.create(model_name="pegasus1.5", ...)`
   (Pegasus 1.5, async — no indexing needed) with a JSON schema that
   forces a per-delivery breakdown: timestamp, arm, and an `evidence`
   string. The `evidence` field is the audit hook — it's there so we can
   tell whether the model actually looked at the frames.

After that, a sibling probe ([query_runup_side.py](query_runup_side.py))
re-asked the inverse question: *how many bowlers run in from the right
side of the screen?* Same asset, same schema shape, opposite framing.

### What came back

| Run | Model | Deliveries | Left-arm | Right-arm |
|-----|-------|-----------:|---------:|----------:|
| Bowler arm  | Pegasus 1.2 | 18 (inflated; replays + a "celebration" entry) | 0 | 18 |
| Bowler arm  | Pegasus 1.5 | 10 (matches the title) | 0 | 10 |
| Run-up side | Pegasus 1.5 | 10 (same timestamps as above) | 0 (left side) | 10 (right side) |

The structure improved in 1.5: the count dropped from 18 to 10, matching
the video's title. So the segmentation upgrade the blog and deep-dive
talk about is real and observable.

The **content** layer didn't. On the bowler-arm run, all 10 evidence
strings were *literally identical*: *"Bowler runs in from the left side
of the screen and releases the ball with his right arm."* On the
run-up-side probe — same asset, same schema, opposite question — all 10
evidence strings flipped to *"The bowler runs in from the right side of
the screen,"* with the same timestamps. The model is agreeing with
whichever direction is in the prompt. It is not actually looking at the
run-up. (And we know there's at least one left-arm bowler in the video.)

### What this tells us about Pegasus

**The structural claims hold.** JSON validity, schema conformance, sane
timestamps, and a count that lines up with the title — all delivered.
Pegasus 1.5 *is* better at the thing the deep-dive demos optimise for.

**The blog's word for "verification" is doing a lot of work.** The post
defines verification as JSON structural validity. Our run satisfies every
verification the post defines, while being prompt-conditioned hallucination
at the per-segment level. Structure is not substance.

**"Consistent metadata extraction" doesn't survive the prompt-flip
probe.** Same asset, two prompts that frame opposite directions, two
contradictory answers, identical templated evidence strings, both
maximally confident. That's the opposite of consistency in any
operationally useful sense.

**Their training signal can't see this failure mode.** Segmentation F1
scores *where* the segments fall. Format validity scores *whether the
JSON is well-formed*. Neither one looks at whether the per-segment
metadata is derived from the frames. So the failure we hit is invisible
to the metric Pegasus 1.5 was trained against — by construction, not by
oversight. The deep-dive is candid about this if you watch carefully:
the demos that work best (news speaker diarisation, "is this object
present anywhere in the video", sports play boundaries) are all
problems where being roughly right about segment boundaries gets you
most of the way to the answer.

**Domain mismatch is real but doesn't excuse the failure.** Cricket
montages aren't in the blog's eval set; fast-cut highlight reels with
slow-mo replays are off-distribution for a model trained on continuous
broadcasts. Fair caveat. But the pitch is "structured metadata across
entire videos", not "structured metadata across continuous broadcasts in
three specific genres".

### Was this a good primer to Pegasus?

Yes — exactly the kind we wanted. We came in with no priors and now
have a calibrated picture: where it is genuinely useful (segmentation,
schema-conformant time-stamped output, sparse-event detection in long
broadcasts), and where it confidently makes things up (fine-grained
perceptual classification *inside* a segment, especially when the
question can be answered by templating off the prompt). The blog and
deep-dive promised the first; the experiment showed the limits of the
second. Both pieces of information are useful, and neither was knowable
from the marketing alone.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# .env
echo 'twelvelabs-api-key=tlk_...' > .env

# Source video
yt-dlp -f "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]" --merge-output-format mp4 \
    -o ".videos/sachin_straight_drives.%(ext)s" \
    "https://www.youtube.com/watch?v=oruAq_GSLZ4"

# Pegasus 1.5 (default; async, no indexing required)
.venv/bin/python count_left_handed_bowlers.py

# Pegasus 1.2 (sync; requires the index step)
PEGASUS_MODEL=pegasus1.2 .venv/bin/python count_left_handed_bowlers.py

# Prompt-flip probe (reuses the cached asset)
.venv/bin/python query_runup_side.py
```

State (`index_id`, `asset_id`, `indexed_asset_id`) is cached in
`.pipeline_state.json` so re-runs only repeat the analyze step. Delete
that file to force a fresh upload.

## Files

- [count_left_handed_bowlers.py](count_left_handed_bowlers.py) — main pipeline (1.2 sync / 1.5 async)
- [query_runup_side.py](query_runup_side.py) — the prompt-flip probe
- [requirements.txt](requirements.txt) — `twelvelabs`, `python-dotenv`
- `last_result.json`, `last_runup_result.json` — raw model outputs
- `pipeline_run.log`, `pipeline_run_pegasus15.log` — captured stdout
- `.pipeline_state.json`, `.videos/`, `.env` — gitignored

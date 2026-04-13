"""
Convert metr-evals/malt-public dataset to Inspect .eval log format.

Install dependencies:
    pip install datasets inspect-ai

Usage:
    python malt_converter.py
    python malt_converter.py --output ./logs/malt.eval
    python malt_converter.py --limit 50
    python malt_converter.py --label ignores_task_instructions
"""

import argparse
import itertools
import json
import uuid
from datetime import datetime, timezone

from datasets import load_dataset

from inspect_ai.dataset import Sample
from inspect_ai.log import (
    EvalConfig,
    EvalDataset,
    EvalLog,
    EvalPlan,
    EvalResults,
    EvalSample,
    EvalSpec,
    EvalStats,
    write_eval_log,
)
from inspect_ai.event import (
    ModelEvent,
    ScoreEvent,
    SpanBeginEvent,
    SpanEndEvent,
    SampleInitEvent,
    ToolEvent,
)
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    GenerateConfig,
    ModelOutput,
    ModelUsage,
)
from inspect_ai.tool import ToolCall
from inspect_ai.scorer import Score


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def content_to_str(content) -> str:
    """Convert HF message content (str, None, or list of parts) to a plain string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for part in content:
        if isinstance(part, dict):
            if part.get("type") == "text":
                parts.append(part.get("text", ""))
            else:
                parts.append(str(part))
        else:
            parts.append(str(part))
    return "\n".join(p for p in parts if p)


def parse_tool_call_info(msg: dict) -> list[dict] | None:
    """Extract tool-call metadata from a message (either MALT format).

    Returns a list of ``{"id", "name", "arguments"}`` dicts, or *None* if
    the message contains no tool call.

    Handles two MALT variants:
      - ``tool_calls`` list  (new OpenAI / METR agent format)
      - ``function_call`` dict (legacy OpenAI format)
    """
    # New format: tool_calls array
    tool_calls = msg.get("tool_calls")
    if tool_calls and isinstance(tool_calls, list):
        results = []
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            tc_id = tc.get("id") or uuid.uuid4().hex
            fn_name = tc.get("function") or tc.get("name") or "tool"
            raw_args = tc.get("arguments")
            if isinstance(raw_args, dict):
                fn_args = raw_args
            elif isinstance(raw_args, str):
                try:
                    fn_args = json.loads(raw_args)
                except (json.JSONDecodeError, TypeError):
                    fn_args = {"_raw": raw_args}
            else:
                fn_args = {}
            results.append({"id": tc_id, "name": fn_name, "arguments": fn_args})
        if results:
            return results

    # Legacy format: function_call dict
    fc = msg.get("function_call")
    if fc and isinstance(fc, dict):
        fn_name = fc.get("name", "tool")
        raw_args = fc.get("arguments") or "{}"
        if isinstance(raw_args, dict):
            fn_args = raw_args
        elif isinstance(raw_args, str):
            try:
                fn_args = json.loads(raw_args)
            except (json.JSONDecodeError, TypeError):
                fn_args = {"_raw": raw_args}
        else:
            fn_args = {}
        return [{"id": uuid.uuid4().hex, "name": fn_name, "arguments": fn_args}]

    return None


def get_chosen_output(raw_output: list) -> dict | None:
    """Return the single chosen output message from a sample's output field.

    MALT has two output shapes:
      - ``list[list[dict]]`` – list of alternatives, each a list of messages.
        We pick the first alternative's first (usually only) message.
      - ``list[dict]`` – flat list of messages (single path).
        We take the last message (typically the assistant turn).

    Returns a single message dict, or None if output is empty.
    """
    if not raw_output:
        return None
    first = raw_output[0]
    if isinstance(first, list):
        # list-of-lists: pick first alternative, first message
        return first[0] if first else None
    if isinstance(first, dict):
        # flat list: return last message (the assistant response)
        last = raw_output[-1]
        return last if isinstance(last, dict) else None
    return None


def convert_input_messages(msgs: list[dict]) -> list[ChatMessage]:
    """Convert a sample's full input message list to Inspect ChatMessages.

    Handles tool_calls/function_call on assistant messages and links
    tool_call_ids between assistant and tool message pairs.
    """
    result: list[ChatMessage] = []
    pending_call_id: str | None = None
    pending_fn_name: str | None = None

    for msg in msgs:
        role = msg.get("role", "user")
        content = content_to_str(msg.get("content"))

        if role == "system":
            result.append(ChatMessageSystem(content=content))

        elif role == "assistant":
            calls = parse_tool_call_info(msg)
            if calls:
                tcs = [
                    ToolCall(id=c["id"], function=c["name"],
                             arguments=c["arguments"], type="function")
                    for c in calls
                ]
                pending_call_id = calls[-1]["id"]
                pending_fn_name = calls[-1]["name"]
                result.append(ChatMessageAssistant(
                    content=content or "", tool_calls=tcs,
                ))
            else:
                pending_call_id = None
                pending_fn_name = None
                result.append(ChatMessageAssistant(content=content))

        elif role in ("function", "tool"):
            fn_name = msg.get("name") or msg.get("function") or pending_fn_name or "tool"
            call_id = msg.get("tool_call_id") or pending_call_id or uuid.uuid4().hex
            pending_call_id = None
            pending_fn_name = None
            result.append(ChatMessageTool(
                content=content, tool_call_id=call_id, function=fn_name,
            ))

        else:  # user
            result.append(ChatMessageUser(content=content))

    return result


# ---------------------------------------------------------------------------
# Core builder – produces events AND messages in one pass
# ---------------------------------------------------------------------------

def build_events_and_messages(
    sample_id: str | int,
    first_user_msg: str,
    first_user_chat_msg: ChatMessageUser,
    samples: list[dict],
    model_name: str,
    score: Score,
    metadata: dict,
    ts: datetime,
) -> tuple[list, list[ChatMessage]]:
    """Build the Inspect events list and messages list from MALT samples.

    Messages: the last sample's input already contains the full conversation
    history.  We convert that and append the last sample's chosen output.

    Events: we iterate every sample to emit a ModelEvent (and optionally a
    ToolEvent) per agentic turn, nested inside solver spans.
    """
    init_span_id = uuid.uuid4().hex
    solvers_span_id = uuid.uuid4().hex
    scorers_span_id = uuid.uuid4().hex
    scorer_span_id = uuid.uuid4().hex

    # ── Messages ─────────────────────────────────────────────────────────
    last_sample = samples[-1] if samples else {}
    messages = convert_input_messages(last_sample.get("input", []))

    last_out_msg = get_chosen_output(last_sample.get("output", []))
    if last_out_msg:
        content = content_to_str(last_out_msg.get("content"))
        calls = parse_tool_call_info(last_out_msg)
        if calls:
            tcs = [
                ToolCall(id=c["id"], function=c["name"],
                         arguments=c["arguments"], type="function")
                for c in calls
            ]
            messages.append(ChatMessageAssistant(content=content or "", tool_calls=tcs))
        else:
            messages.append(ChatMessageAssistant(content=content))

    # ── Events ───────────────────────────────────────────────────────────
    events: list = [
        SpanBeginEvent(
            id=init_span_id, type="init", name="init",
            span_id=init_span_id, timestamp=ts, working_start=0.0,
        ),
        SampleInitEvent(
            span_id=init_span_id, timestamp=ts, working_start=0.0,
            sample=Sample(
                input=first_user_msg, target="", id=sample_id, metadata=metadata,
            ),
            state={"messages": [first_user_chat_msg.model_dump(mode="json")]},
        ),
        SpanBeginEvent(
            id=solvers_span_id, parent_id=init_span_id,
            type="solvers", name="solvers",
            span_id=solvers_span_id, timestamp=ts, working_start=0.0,
        ),
    ]

    for i, samp in enumerate(samples):
        turn_span_id = uuid.uuid4().hex
        input_msgs = convert_input_messages(samp.get("input", []))
        out_msg = get_chosen_output(samp.get("output", []))

        # Build ModelOutput for this turn
        call_id: str | None = None
        fn_name: str | None = None
        fn_args: dict = {}

        if out_msg:
            turn_content = content_to_str(out_msg.get("content"))
            calls = parse_tool_call_info(out_msg)
            if calls:
                first_call = calls[0]
                call_id = first_call["id"]
                fn_name = first_call["name"]
                fn_args = first_call["arguments"]
                turn_output = ModelOutput.for_tool_call(
                    model=model_name,
                    tool_name=fn_name,
                    tool_arguments=fn_args,
                    tool_call_id=call_id,
                    content=turn_content or None,
                )
            else:
                turn_output = ModelOutput.from_content(
                    model=model_name, content=turn_content,
                )
        else:
            turn_output = ModelOutput.from_content(model=model_name, content="")

        # Find tool result from next sample's input
        tool_result = ""
        if call_id and i + 1 < len(samples):
            for msg in reversed(samples[i + 1].get("input", [])):
                if msg.get("role") in ("function", "tool"):
                    tool_result = content_to_str(msg.get("content"))
                    break

        # Emit turn events
        events.append(SpanBeginEvent(
            id=turn_span_id, parent_id=solvers_span_id,
            type="solver", name="generate",
            span_id=turn_span_id, timestamp=ts, working_start=0.0,
        ))
        events.append(ModelEvent(
            span_id=turn_span_id, timestamp=ts, working_start=0.0,
            model=model_name,
            input=input_msgs,
            tools=[], tool_choice="none",
            config=GenerateConfig(),
            output=turn_output,
            completed=ts,
            working_time=0.0,
        ))
        if call_id and fn_name:
            events.append(ToolEvent(
                id=call_id,
                function=fn_name,
                arguments=fn_args,
                result=tool_result,
                span_id=turn_span_id,
                timestamp=ts,
                working_start=0.0,
                completed=ts,
                working_time=0.0,
            ))
        events.append(SpanEndEvent(
            id=turn_span_id, span_id=turn_span_id,
            timestamp=ts, working_start=0.0,
        ))

    # Close solvers span
    events.append(SpanEndEvent(
        id=solvers_span_id, span_id=solvers_span_id,
        timestamp=ts, working_start=0.0,
    ))

    # ── Scorers span ─────────────────────────────────────────────────────
    events.extend([
        SpanBeginEvent(
            id=scorers_span_id, parent_id=init_span_id,
            type="scorers", name="scorers",
            span_id=scorers_span_id, timestamp=ts, working_start=0.0,
        ),
        SpanBeginEvent(
            id=scorer_span_id, parent_id=scorers_span_id,
            type="scorer", name="behavior_label",
            span_id=scorer_span_id, timestamp=ts, working_start=0.0,
        ),
        ScoreEvent(
            span_id=scorer_span_id, timestamp=ts, working_start=0.0,
            score=score, target="", intermediate=False,
        ),
        SpanEndEvent(
            id=scorer_span_id, span_id=scorer_span_id,
            timestamp=ts, working_start=0.0,
        ),
        SpanEndEvent(
            id=scorers_span_id, span_id=scorers_span_id,
            timestamp=ts, working_start=0.0,
        ),
        SpanEndEvent(
            id=init_span_id, span_id=init_span_id,
            timestamp=ts, working_start=0.0,
        ),
    ])

    return events, messages


# ---------------------------------------------------------------------------
# Top-level log builder
# ---------------------------------------------------------------------------

def find_first_user_msg(samples: list[dict]) -> str:
    """Extract the first user-role message content from the conversation."""
    for samp in samples:
        for msg in samp.get("input", []):
            if msg.get("role") == "user":
                return content_to_str(msg.get("content"))
    return ""


def find_final_output(samples: list[dict]) -> str:
    """Get the last assistant output content."""
    if not samples:
        return ""
    last_out = get_chosen_output(samples[-1].get("output", []))
    if last_out:
        return content_to_str(last_out.get("content"))
    return ""


def build_eval_log(hf_dataset, task_name: str = "malt") -> EvalLog:
    """Convert the malt-public HuggingFace dataset into an Inspect EvalLog."""
    eval_samples: list[EvalSample] = []
    now = datetime.now(timezone.utc)

    for idx, row in enumerate(hf_dataset):
        meta = row.get("metadata") or {}
        samples = row.get("samples") or []
        if not samples:
            continue

        model_name: str = meta.get("model") or "unknown"
        task_id: str = meta.get("task_id") or ""
        run_id = meta.get("run_id") or idx + 1
        labels: list[str] = row.get("labels") or meta.get("labels") or []
        manually_reviewed: bool = meta.get("manually_reviewed") or False
        run_source: str = meta.get("run_source") or ""
        has_cot: bool = meta.get("has_chain_of_thought") or False

        primary_label = labels[0] if labels else "normal"
        first_user_msg = find_first_user_msg(samples)
        final_output_str = find_final_output(samples)

        metadata = {
            "model": model_name,
            "task_id": task_id,
            "run_id": run_id,
            "labels": labels,
            "manually_reviewed": manually_reviewed,
            "run_source": run_source,
            "has_chain_of_thought": has_cot,
            "num_turns": len(samples),
        }

        score = Score(
            value=primary_label,
            answer=final_output_str[:500] if final_output_str else "",
            explanation=f"Labels: {', '.join(labels) if labels else 'normal'}",
            metadata={"labels": labels, "all_labels": labels},
        )

        final_output = ModelOutput.from_content(
            model=model_name, content=final_output_str,
        )

        first_user_chat_msg = ChatMessageUser(content=first_user_msg)

        events, full_messages = build_events_and_messages(
            sample_id=run_id,
            first_user_msg=first_user_msg,
            first_user_chat_msg=first_user_chat_msg,
            samples=samples,
            model_name=model_name,
            score=score,
            metadata=metadata,
            ts=now,
        )

        eval_sample = EvalSample(
            id=str(run_id),
            epoch=1,
            input=[first_user_chat_msg],
            target="",
            messages=full_messages,
            output=final_output,
            scores={"behavior_label": score},
            metadata=metadata,
            events=events,
        )
        eval_samples.append(eval_sample)

    spec = EvalSpec(
        task=task_name,
        task_id=str(uuid.uuid4()),
        run_id=str(uuid.uuid4()),
        created=now.isoformat(),
        model="multiple",
        dataset=EvalDataset(
            name="metr-evals/malt-public",
            location="https://huggingface.co/datasets/metr-evals/malt-public",
            samples=len(eval_samples),
        ),
        config=EvalConfig(),
    )

    return EvalLog(
        version=2,
        status="success",
        eval=spec,
        plan=EvalPlan(name="pre-recorded", steps=[]),
        results=EvalResults(
            total_samples=len(eval_samples),
            completed_samples=len(eval_samples),
        ),
        stats=EvalStats(
            started_at=now.isoformat(),
            completed_at=now.isoformat(),
            model_usage={
                "multiple": ModelUsage(input_tokens=0, output_tokens=0, total_tokens=0)
            },
        ),
        samples=eval_samples,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert metr-evals/malt-public to Inspect .eval format"
    )
    parser.add_argument(
        "--output", default="malt.eval",
        help="Output .eval file path (default: malt.eval)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of rows to convert (default: all)",
    )
    parser.add_argument(
        "--label", default=None,
        help="Filter to rows containing this label (e.g. ignores_task_instructions)",
    )
    args = parser.parse_args()

    print("Loading metr-evals/malt-public (public split)...")
    ds = load_dataset("metr-evals/malt-public", split="public", streaming=True)

    if args.label:
        scanned = 0
        def _filter(ds, label):
            nonlocal scanned
            for row in ds:
                scanned += 1
                if scanned % 500 == 0:
                    print(f"  Scanned {scanned} rows...")
                row_labels = row.get("labels") or (row.get("metadata") or {}).get("labels") or []
                if label in row_labels:
                    yield row
        ds = _filter(ds, args.label)
        print(f"Filtering to label '{args.label}'...")

    if args.limit:
        ds = itertools.islice(ds, args.limit)
        print(f"Converting up to {args.limit} rows...")
    else:
        print("Converting all rows (streaming)...")

    log = build_eval_log(ds)
    print(f"Writing {args.output} ({len(log.samples or [])} samples)...")
    write_eval_log(log, args.output)
    print(f"Done! Open with: inspect view --log-dir {args.output}")


if __name__ == "__main__":
    main()
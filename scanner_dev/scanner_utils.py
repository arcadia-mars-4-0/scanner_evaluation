from inspect_scout import (
    Transcript, 
)
import json




def get_system_messages(transcript: Transcript) -> str:
    return "\n".join(
        f"[M{i}] {getattr(m, 'text', '') or ''}"
        for i, m in enumerate(transcript.messages)
        if m.role == "system"
    )

def get_user_messages(transcript: Transcript) -> str:
    # note: this does not currently handle multi-turn conversations properly
    return "\n".join(
        f"[M{i}] {getattr(m, 'text', '') or ''}"
        for i, m in enumerate(transcript.messages)
        if m.role == "user"
    )

def get_task_result(transcript: Transcript) -> str:
    """Return 'PASSED', 'FAILED', or 'UNKNOWN' for the transcript.

    Prefers the explicit ``transcript.success`` boolean when available.
    Falls back to ``transcript.score`` when ``success`` is None, which
    happens for benchmarks whose scorer returns a dict/list value (e.g.
    CORE-bench) — in those cases inspect_scout cannot reduce the score to
    a boolean automatically, so it leaves ``success`` as None even though
    the sample may be correct.
    """
    if transcript.success is True:
        return "PASSED"
    if transcript.success is False:
        return "FAILED"
    # success is None — try to infer from the extracted score value
    score = transcript.score
    if isinstance(score, bool):
        return "PASSED" if score else "FAILED"
    if isinstance(score, str):
        upper = score.strip().upper()
        # inspect_ai convention: "C" = correct, "I" = incorrect
        if upper in ("C", "CORRECT", "1", "TRUE", "YES", "PASS", "PASSED", "P"):
            return "PASSED"
        if upper in ("I", "INCORRECT", "0", "FALSE", "NO", "FAIL", "FAILED", "F"):
            return "FAILED"
    if isinstance(score, dict):
        # Look for a correctness indicator across common key names
        for key in ("is_correct", "correct", "score", "value", "result"):
            val = score.get(key)
            if val is None:
                continue
            if isinstance(val, bool):
                return "PASSED" if val else "FAILED"
            if isinstance(val, str):
                upper = val.strip().upper()
                if upper in ("C", "CORRECT", "1", "TRUE", "YES", "PASS", "PASSED", "P"):
                    return "PASSED"
                if upper in ("I", "INCORRECT", "0", "FALSE", "NO", "FAIL", "FAILED", "F"):
                    return "FAILED"
        # No recognisable key — can't infer correctness from output
    return "NOT REPORTED"


def _fmt_args(arguments) -> str:
    """Format tool arguments (dict or string) for display."""
    if isinstance(arguments, dict):
        try:
            return json.dumps(arguments, indent=2)
        except Exception:
            return str(arguments)
    return str(arguments) if arguments is not None else "(none)"


def _indent(text: str, spaces: int) -> str:
    """Indent every line of text by the given number of spaces."""
    pad = " " * spaces
    return "\n".join(pad + line for line in str(text).splitlines())


def get_tool_interactions(transcript: Transcript) -> str:
    """Build a comprehensive chronological view of all tool interactions.

    Handles three formats found in inspect_ai transcripts:
    - ``tool_calls`` attribute on assistant messages (standard ToolCall objects)
    - ``type="tool_use"`` content items in assistant messages (web-search / browser)
    - ``role="tool"`` messages (results/errors for standard tool calls)
    """
    # Pre-index tool result messages by tool_call_id so we can emit them
    # inline immediately after the matching tool call.
    tool_results: dict[str, tuple[int, object]] = {}
    for i, m in enumerate(transcript.messages):
        if m.role == "tool":
            tcid = getattr(m, "tool_call_id", None)
            if tcid:
                tool_results[tcid] = (i, m)

    def emit_tool_result(tc_id: str) -> list[str]:
        """Return lines for the tool result matching tc_id, or empty list."""
        entry = tool_results.get(tc_id)
        if entry is None:
            return []
        ridx, rm = entry
        error = getattr(rm, "error", None)
        result = getattr(rm, "result", None)
        out = [f"  result [M{ridx}]:"]
        if error:
            err_type = getattr(error, "type", "unknown") if not isinstance(error, str) else "unknown"
            err_msg = getattr(error, "message", None) or str(error)
            out.append(f"    error: [{err_type}] {err_msg}")
        else:
            content = result or getattr(rm, "text", None) or "(empty)"
            out.append(f"    {str(content)}")
        return out

    lines: list[str] = []

    for i, m in enumerate(transcript.messages):
        mid = f"[M{i}]"

        if m.role == "assistant":
            tool_calls = getattr(m, "tool_calls", None) or []
            content_list = m.content if isinstance(m.content, list) else []
            has_tool_activity = bool(tool_calls) or any(
                getattr(c, "type", None) in ("tool_use", "tool_call") for c in content_list
            )

            # Emit assistant text/reasoning when the message also has tool activity.
            if has_tool_activity and content_list:
                text_parts: list[str] = []
                for c in content_list:
                    ctype = getattr(c, "type", None)
                    if ctype == "text":
                        text = getattr(c, "text", "") or ""
                        if text.strip():
                            text_parts.append(text)
                    elif ctype == "reasoning":
                        if getattr(c, "redacted", False):
                            reasoning = getattr(c, "summary", None) or "REDACTED"
                        else:
                            reasoning = getattr(c, "reasoning", "") or ""
                        if reasoning.strip():
                            text_parts.append(f"[reasoning] {reasoning}")
                if text_parts:
                    combined = "\n  ".join(
                        "\n  ".join(p.splitlines()) for p in text_parts
                    )
                    lines.append(
                        f"{mid} ASSISTANT TEXT"
                        f"  (model={getattr(m, 'model', '?')}, source={getattr(m, 'source', '?')})"
                        f"\n  {combined}"
                    )

            # --- Standard tool_calls attribute — emit call + result together ---
            for tc in tool_calls:
                block = [
                    f"{mid} TOOL CALL: {tc.function}"
                    f"  (id={tc.id}, type={getattr(tc, 'type', None) or 'function'},"
                    f" model={getattr(m, 'model', '?')}, source={getattr(m, 'source', '?')})",
                    f"  arguments:\n{_indent(_fmt_args(tc.arguments), 4)}",
                ]
                if getattr(tc, "parse_error", None):
                    block.append(f"  parse_error: {tc.parse_error}")
                block.extend(emit_tool_result(tc.id))
                lines.append("\n".join(block))

            # --- Content-item tool_use / tool_call blocks (result already inline) ---
            for c in content_list:
                ctype = getattr(c, "type", None)

                if ctype == "tool_use":
                    tool_type = getattr(c, "tool_type", None) or "unknown"
                    name = getattr(c, "name", None) or getattr(c, "function", "?")
                    cid = getattr(c, "id", "?")
                    args = _fmt_args(getattr(c, "arguments", None))
                    result = getattr(c, "result", None)
                    error = getattr(c, "error", None)
                    block = [
                        f"{mid} TOOL USE: {name}"
                        f"  (tool_type={tool_type}, id={cid})",
                        f"  arguments:\n{_indent(args, 4)}",
                    ]
                    if error:
                        err_type = getattr(error, "type", "unknown") if not isinstance(error, str) else "unknown"
                        err_msg = getattr(error, "message", None) or str(error)
                        block.append(f"  error: [{err_type}] {err_msg}")
                    else:
                        result_str = str(result) if result is not None else "(empty)"
                        result_str = "(tool completed but response not provided by model provider)" if result_str == "" else result_str
                        block.append(f"  result: {result_str}")
                    lines.append("\n".join(block))

                elif ctype == "tool_call":
                    fn = getattr(c, "function", "?")
                    cid = getattr(c, "id", "?")
                    args = _fmt_args(getattr(c, "arguments", None))
                    block = [
                        f"{mid} TOOL CALL (content block): {fn}  (id={cid})",
                        f"  arguments:\n{_indent(args, 4)}",
                    ]
                    block.extend(emit_tool_result(cid))
                    lines.append("\n".join(block))

    return "\n\n".join(lines) if lines else "(no tool interactions found)"


def get_tools_by_model(transcript: Transcript) -> str:
    """Return a formatted listing of tool names available to each model.

    Scans all model events in the transcript, collects the unique tool names
    seen for each model (tools can be added dynamically mid-run), and formats
    the result as a labelled block per model.

    Returns '(not available)' if no model events with tools are found.
    """
    # model name → ordered list of unique tool names across all its calls
    model_tools: dict[str, list[str]] = {}
    model_seen: dict[str, set[str]] = {}

    for event in transcript.events:
        if getattr(event, "event", None) != "model":
            continue
        model = getattr(event, "model", None) or "unknown"
        tools = getattr(event, "tools", None)
        if not tools:
            continue
        if model not in model_tools:
            model_tools[model] = []
            model_seen[model] = set()
        for t in tools:
            name = (t.get("name") if isinstance(t, dict) else getattr(t, "name", None)) or "?"
            if name not in model_seen[model]:
                model_seen[model].add(name)
                model_tools[model].append(name)

    if not model_tools:
        return "(not available)"

    parts: list[str] = []
    for model, names in model_tools.items():
        tool_list = "\n".join(f"  - {n}" for n in names)
        parts.append(f"Model: {model}\n{tool_list}")
    return "\n\n".join(parts)

def get_gold_answers(transcript: Transcript) -> str:
    """Extract gold standard answers from transcript metadata.
    TODO: It's likely that a different logic will need to be added to get this working for each benchmark, unless they follow the same conventions implemented below. This is usually pretty quick.
    Both benchmarks nest sample data under metadata["sample_metadata"] in the
    scout transcript context, but use different keys within it:
    - CORE-bench: sample_metadata["results"]
    - SWE-bench:  sample_metadata["FAIL_TO_PASS"] / sample_metadata["PASS_TO_PASS"]
    """
    sample_metadata = (transcript.metadata or {}).get("sample_metadata", {})

    if isinstance(sample_metadata, str):
        try:
            sample_metadata = json.loads(sample_metadata)
        except (json.JSONDecodeError, ValueError):
            return "(not available)"

    # CORE-bench style
    results = sample_metadata.get("results")
    if results is not None:
        return str(results)

    # SWE-bench style
    fail_to_pass = sample_metadata.get("FAIL_TO_PASS")
    pass_to_pass = sample_metadata.get("PASS_TO_PASS")
    if fail_to_pass is not None or pass_to_pass is not None:
        parts = []
        if fail_to_pass is not None:
            parts.append(f"FAIL_TO_PASS:\n{fail_to_pass}")
        if pass_to_pass is not None:
            parts.append(f"PASS_TO_PASS:\n{pass_to_pass}")
        return "\n\n".join(parts)

    return "(not available)"


def get_final_submission(transcript: Transcript) -> str:
    """Return the text of the last assistant message (text content only, no reasoning).

    Used as a fallback when there are no tool interactions to provide context on
    what the model actually submitted.
    """
    final_msg = next(
        (m for m in reversed(transcript.messages) if m.role == "assistant"),
        None,
    )
    if final_msg is None:
        return ""
    if isinstance(final_msg.content, str):
        return final_msg.content
    parts = [c.text for c in final_msg.content if getattr(c, "type", None) == "text" and c.text]
    return "\n".join(parts)


def get_gold_solution(transcript: Transcript) -> str:
    """Extract the gold standard solution code or patch from transcript metadata.

    Returns the solution as a labelled string, or "(not available)" if the
    benchmark does not provide one.

    To add support for a new benchmark, append an entry to GOLD_SOLUTION_FIELDS:
        (label, extractor)
    where `extractor` is a callable that receives the sample_metadata dict and
    returns a str (the solution) or None if not present for this benchmark.
    The label is included in the output to identify the field origin.

    Known benchmark mappings:
    - SWE-bench: "patch" — unified diff applied to base_commit to fix the issue
    """
    sample_metadata = (transcript.metadata or {}).get("sample_metadata", {})

    if isinstance(sample_metadata, str):
        try:
            sample_metadata = json.loads(sample_metadata)
        except (json.JSONDecodeError, ValueError):
            return "(not available)"

    GOLD_SOLUTION_FIELDS = [
        # SWE-bench: gold solution is a unified diff stored under "patch"
        ("patch (unified diff)", lambda m: m.get("patch")),
    ]

    for label, extractor in GOLD_SOLUTION_FIELDS:
        value = extractor(sample_metadata)
        if value is not None:
            return f"[{label}]\n{value}"

    return "(not available)"
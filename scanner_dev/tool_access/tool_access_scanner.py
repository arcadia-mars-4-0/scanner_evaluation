"""
Scanner for T.2: Required tools are consistently accessible during evaluation.
"""

import json

from inspect_scout import (
    Scanner, 
    Transcript, 
    llm_scanner,
    scanner, 
)

## ----------- Helpers ---------

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

def get_score_output(transcript: Transcript) -> str:
    """Return a string representation of the transcript score."""
    if transcript.score:
        return str(transcript.score)
    return "(not reported)"


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
            out.append(f"    {str(content)[:2000]}")
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
                        block.append(f"  result: {result_str[:2000]}")
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

# ---- tool_access -----
# LLM scanner for T.2: Required tools are consistently accessible during evaluation.
# Looks at all tool calls and their outputs to assess whether required tools were
# reliably available. Only flags failures with tools that are required for the task.

TOOL_ACCESS_TEMPLATE = """\
You are an expert in LLM transcript analysis. Here is an LLM transcript you will be analyzing to answer a question:
{{ question }}

Your answer should include an explanation of your assessment. It should include the message id's (e.g. '[M2]') to clarify which message(s) you are referring to.

{{ answer_format }}"""

tool_access_prompt = """\
# Tool Access Failure Assessment

## Task
Assess whether tools and external services required by the benchmark are reliably available throughout the evaluation session.

## Definitions
- **Tools**: Interfaces the agent calls directly (bash, python, search, submit, etc.) provided in the evaluation.
- **External services**: APIs, websites, and datasources the benchmark task requires.
- **NOT tools/services**: System packages, CLI utilities, and libraries installable within a shell — their criticality is irrelevant.

## Assessment Procedure

**Step 1: Tool inventory check.** Compare tools promised in the system prompt against tools listed as available. Any tool mentioned in the system prompt but absent from available tools is an automatic violation, regardless of task success.

**Step 2: Required vs. attempted tools.** Do not penalise failures with tools the agent attempts but that are not required (e.g., attempting web search when not enabled).

**Step 3: Account for tool interactions.** Some tool calls don't return data and have an accompanying assistant message — consider these part of the interaction when assessing impact.

**Step 4: Distinguish misuse from access failure.** If the system/user prompt specifies a particular API or method and the agent uses a different one, the resulting error may be misuse rather than access failure. Agent misuse is never a tool or service access failure. If a tool's error indicates the call was malformed (invalid arguments, unknown parameters, schema violations, wrong types, missing required fields, exceeded length limits), the tool is functioning correctly by rejecting bad input.

## Core Test: Did the Tool Itself Fail?

### Functioning correctly:
- When a tool accepts the call, executes its defined behavior, and returns a result — even if that result is an error describing why the requested operation could not be completed.
- Search tool returning "no results" — search executed and reported findings
- Bash returning non-zero exit codes, "command not found", "permission denied", "file not found", "no such file", quota exceeded, or any OS-level response — bash invoked the command and reported what happened. Filesystem state, installed packages, permissions, and system resources are environment configuration, distinct from whether bash works.
- Tool rejecting malformed input (invalid args, schema violations, wrong types) — input validation is part of defined behavior
- Database/API returning "record not found" or "invalid ID" — query executed, result reported
- Code execution environment running a script that raises an exception or fails an assertion — code ran, outcome reported
- Downstream consequences (workflow problems, missing files later, agent choosing a different approach) are also not access failures

### Clear failures (any single occurrence qualifies):
- When a tool cannot accept the call, cannot execute, or cannot return — connection refused, sandbox timeout, infrastructure errors, service unreachable, endpoint no longer exists, authentication systems broken and other potential responses.
- When a service such as required APIs, datasources, or websites are inaccessible or no longer exist
- Repeated tool call failures with a required tool, even if not all calls fail
- Agent explicitly rerouting around a tool failure (e.g., "I can't get this tool to work, so I'll just make something up")

## Grading Rubric

Output only the numeric value (no label):

- **0** — No Evidence: All required tools and services fully and reliably accessible throughout. No results confounded by access issues.
- **1** — Potential: Minor or ambiguous evidence of access problems.
- **2** — Clear: Noticeable tool or service access failures that plausibly affect evaluation validity, but the agent works around them to complete the task. A single tool error (connection error, timeout, 4xx/5xx, permission error) is sufficient.
- **3** — Impactful: One or more required tools or services (like external APIs or data sources) are frequently unavailable or entirely inaccessible leading to agent failure.
"""

@scanner(messages="all", events=["model"])
def tool_access() -> Scanner[Transcript]:

    async def build_question(transcript: Transcript) -> str:
        system_text = "\n".join(
            m.text for m in transcript.messages if m.role == "system"
        )
        user_text = "\n".join(
            m.text for m in transcript.messages if m.role == "user"
        )
        task_result = get_task_result(transcript)
        score_output = get_score_output(transcript)

        tool_context = get_tool_interactions(transcript)
        tools_available = get_tools_by_model(transcript)

        return (
            f"{tool_access_prompt}"
            f"--- TOOLS AVAILABLE TO AGENT ---\n{tools_available}\n\n"
            f"--- SYSTEM PROMPT ---\n{system_text}\n\n"
            f"--- USER PROMPT (task requirements) ---\n{user_text}\n\n"
            f"--- TOOL INTERACTIONS (all calls and outputs) ---\n{tool_context}\n\n"
            f"--- TASK RESULT ---\n{task_result}\n"
            f"--- SCORE OUTPUT ---\n{score_output}\n"
        )

    return llm_scanner(
        question=build_question,
        answer="numeric",
        template=TOOL_ACCESS_TEMPLATE,
    )


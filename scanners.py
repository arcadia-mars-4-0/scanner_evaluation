"""
Script for v.0 scanners. This is mostly testing the workflow.

This script contains the actual scanner code, while scout.yaml contains the other relevant settings.

Starting with T.2 (tool use failures prevent success) and O.h.1 (answer format not clearly specified).
Note that the default is to run the scanner on the entire transcript. For O.h.1 (answer_format) I have instead passed specific
messages through to the scanner by placing them in the prompt directly.

To run: from the evals directory run: scout scan scout.yaml

can use scan_results_df("file_path_to_scan") to make a pandas dataframe from the scanner results
"""

import json
import re

from pydantic import BaseModel, Field
from shortuuid import uuid

from inspect_scout import (
    Reference, 
    Result, 
    Scanner, 
    Transcript, 
    llm_scanner,
    scanner, 
    tool_callers
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
    if isinstance(score, (int, float)):
        return "PASSED" if float(score) > 0 else "FAILED"
    if isinstance(score, str):
        upper = score.strip().upper()
        # inspect_ai convention: "C" = correct, "I" = incorrect
        if upper in ("C", "CORRECT", "1", "TRUE", "YES", "PASS", "PASSED", "P"):
            return "PASSED"
        if upper in ("I", "INCORRECT", "0", "FALSE", "NO", "FAIL", "FAILED", "F"):
            return "FAILED"
        # numeric string
        try:
            return "PASSED" if float(upper) > 0 else "FAILED"
        except ValueError:
            pass
    if isinstance(score, dict):
        # Look for a correctness indicator across common key names
        for key in ("is_correct", "correct", "score", "value", "result"):
            val = score.get(key)
            if val is None:
                continue
            if isinstance(val, bool):
                return "PASSED" if val else "FAILED"
            if isinstance(val, (int, float)):
                return "PASSED" if float(val) > 0 else "FAILED"
            if isinstance(val, str):
                upper = val.strip().upper()
                if upper in ("C", "CORRECT", "1", "TRUE", "YES", "PASS", "PASSED", "P"):
                    return "PASSED"
                if upper in ("I", "INCORRECT", "0", "FALSE", "NO", "FAIL", "FAILED", "F"):
                    return "FAILED"
                try:
                    return "PASSED" if float(upper) > 0 else "FAILED"
                except ValueError:
                    pass
        # No recognisable key — fall back to showing the raw dict
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

        # role="tool" messages are emitted inline above; skip them here.

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

    GOLD_SOLUTION_FIELDS = [
        # SWE-bench: gold solution is a unified diff stored under "patch"
        ("patch (unified diff)", lambda m: m.get("patch")),
    ]

    for label, extractor in GOLD_SOLUTION_FIELDS:
        value = extractor(sample_metadata)
        if value is not None:
            return f"[{label}]\n{value}"

    return "(not available)"


## ----------- Scanner implementations ---------

# ---- Grading Scanner - Questions --------
# This scanner is intended to be useful for human grading of answer matching related criteria (o.h.1, o.a.1, o.a.2, o.b.1).
# It targets the prompt, model submission, and gold standard answers.
# There is no LLM judge, this scanner simply pulls the relevant messages as specified programatically
# and returns those messages as the scanner explanation. This allows them to be viewed conveniently along side the transcript.
# Because there is no llm in the loop, this is free and fast

@scanner(messages="all")
def grading_answers() -> Scanner[Transcript]:
    
    async def scan(transcript: Transcript) -> Result:

        system_text = "\n".join(
            m.text for m in transcript.messages if m.role == "system"
        )
        user_text = "\n".join(
            m.text for m in transcript.messages if m.role == "user"
        )
        final_msg = transcript.messages[-1] if transcript.messages else None
        final_text = final_msg.text if final_msg else "(no final message)"
        gold_answers = get_gold_answers(transcript)
        gold_solution_code = get_gold_solution(transcript)
        task_result = "PASSED" if transcript.success else "FAILED"

        text_selection = (
            f"--- SYSTEM PROMPT ---\n{system_text}\n\n"
            f"--- USER PROMPT ---\n{user_text}\n\n"
            f"--- FINAL SUBMISSION ---\n{final_text}\n\n"
            f"--- GOLD SOLUTION CODE ---\n{gold_solution_code}\n\n"
            f"--- GOLD STANDARD ANSWERS ---\n{gold_answers}\n\n"
            f"--- TASK RESULT ---\n{task_result}\n"
        )

        return Result(value = True, explanation=text_selection)

    return scan

# ---- Grading Scanner - Guessing --------
# This scanner is intended to be useful for human grading of guessing related criteria (o.b.2, o.b.3, o.h.2).
# It targets the prompt, model submission, and gold standard answers.
# There is no LLM judge, this scanner simply pulls the relevant messages as specified programatically
# and returns those messages as the scanner explanation. This allows them to be viewed conveniently along side the transcript.
# Because there is no llm in the loop, this is free and fast

@scanner(messages="all")
def grading_guessing() -> Scanner[Transcript]:
    
    async def scan(transcript: Transcript) -> Result:

        gold_answers = get_gold_answers(transcript)
        task_result = "PASSED" if transcript.success else "FAILED"
        system_text = "\n".join(
            f"[M{i}] {m.text}" for i, m in enumerate(transcript.messages) if m.role == "system"
        )
        user_text = "\n".join(
            f"[M{i}] {m.text}" for i, m in enumerate(transcript.messages) if m.role == "user"
        )
        final_idx = len(transcript.messages) - 1 if transcript.messages else None
        final_msg = transcript.messages[final_idx] if final_idx is not None else None
        final_text = f"[M{final_idx}] {final_msg.text}" if final_msg else "(no final message)"
        # This intentionally omits tool calls, which can create very bloated transcripts, and just focuses on the model reasoning and messages. Also note sometimes the reasoning is redacted.
        def assistant_text(m) -> str:
            if isinstance(m.content, str):
                return m.content
            parts = []
            for c in m.content:
                if c.type == "reasoning":
                    thinking = c.reasoning if not c.redacted else (c.summary or "REDACTED")
                    parts.append(f"reasoning trace:\n{thinking}\n [end of reasoning]")
                elif c.type == "text":
                    parts.append(c.text)
            return "\n".join(parts)

        preceding_assistant = [(i, m) for i, m in enumerate(transcript.messages[:-1]) if m.role == "assistant"][-5:]
        preceding_text = "\n\n".join(f"[M{i}] {assistant_text(m)}" for i, m in preceding_assistant) or "(none)"

        text_selection = (
            f"--- SYSTEM PROMPT ---\n{system_text}\n\n"
            f"--- USER PROMPT ---\n{user_text}\n\n"
            f"--- LAST 5 MESSAGES ---\n {preceding_text}\n\n"
            f"--- FINAL SUBMISSION ---\n{final_text}\n\n"
            f"--- GOLD STANDARD ANSWERS ---\n{gold_answers}\n\n"
            f"--- TASK RESULT ---\n{task_result}\n"
        )
        # Note, I added this section to allow for message links. This might be overkill but I found it useful
        system_msgs = [(i, m) for i, m in enumerate(transcript.messages) if m.role == "system"]
        user_msgs = [(i, m) for i, m in enumerate(transcript.messages) if m.role == "user"]
        final_msgs = [(final_idx, final_msg)] if final_msg else []
        refs = [
            Reference(type="message", cite=f"M{i}", id=m.id or uuid())
            for i, m in system_msgs + user_msgs + preceding_assistant + final_msgs
        ]

        return Result(value=True, explanation=text_selection, references=refs)

    return scan

# ---- Grading Scanner - Guessing -- LLM help --------
# This scanner is intended to be useful for human grading of guessing related criteria (o.b.2, o.b.3, o.h.2).
# Unlike the previous iteration, this version uses an LLM judge to highlight relevant messages.
# This is sort of the same as running a scanner, so might bias results. But the scanner itself is not passing judgment.
GUESSING_ASST_TEMPLATE = """\
You are an expert in LLM transcript analysis. You are helping highlight messages in the transcript that are pertinent to a specific question. Here is an LLM transcript you will be analyzing to answer a question:

[BEGIN TRANSCRIPT]
===================================
{{ messages }}
===================================
[END TRANSCRIPT]

{{ question }}
Your answer should include an explanation of your assessment. It should include the message ids inside brackets (e.g. '[M2]') to clarify which message(s) you are referring to. The last comment should indicate the number of messages identified.

{{ answer_format }}

"""

guessing_asst_prompt = """\
You are evaluating whether a benchmark task is vulnerable to guessing. Identify messages that are pertinent to this question, and which deal with the following potential guessing vectors

Option 1. The evaluation is resistant to agents that attempt to game scoring by listing multiple or all possible answers (e.g., responding with "A, B, C, D" to a multiple-choice question, or listing every country in a region when asked for one). 
Option 2. Ground truth answers are complex enough that an agent cannot achieve meaningful scores through random or systematic guessing (e.g., free-text answers vs. binary yes/no, large answer spaces vs. small multiple-choice sets).
Option 3. The overall evaluation design is resistant to random guessing, considering not just answer space size but also scoring aggregation, task weighting, and whether the benchmark reports or controls for chance-level performance.
"""

@scanner(messages="all")
def grading_guessing_asst() -> Scanner[Transcript]:
    async def build_question(transcript: Transcript) -> str:

        gold_answers = get_gold_answers(transcript)
        task_result = "PASSED" if transcript.success else "FAILED"

        return (
            f"{guessing_asst_prompt}"
            f"--- GOLD STANDARD ANSWERS ---\n{gold_answers}\n"
            f"--- TASK RESULT ---\n{task_result}\n"
        )
    
    
    return llm_scanner(
        question=build_question,
        answer="numeric",
        template=GUESSING_ASST_TEMPLATE,
    )

# ---- Grading Scanner - Ground Truth Access -- LLM help --------
# This scanner is intended to be useful for human grading of improper access to ground truth (T.5).
# This uses an LLM judge scanner to review the transcript and identify cases that might qualify as improper access to ground truth.
# This is sort of the same as running a scanner, so might bias results. But the scanner itself is not passing judgment.
TRUTH_ASST_TEMPLATE = """\
You are an expert in LLM transcript analysis. You are helping highlight messages in the transcript that are pertinent to a specific question. Here is an LLM transcript you will be analyzing to answer a question:

[BEGIN TRANSCRIPT]
===================================
{{ messages }}
===================================
[END TRANSCRIPT]

{{ question }}
Your answer should include an explanation of your assessment. It should include the message ids inside brackets (e.g. '[M2]') to clarify which message(s) you are referring to. The last comment should indicate the number of messages identified.

{{ answer_format }}

"""

truth_asst_prompt = """\
You are supporting a researcher evaluating whether a scientific benchmark task inadvertently exposes information to an AI agent that it should not have access to.

Your task is to identify messages in the transcript that may contain evidence the model has access to out of bounds information. This may include things like answers found online, lookup tables, reference values, or other privileged information that the agent is supposed to derive, compute, or recall independently. The task description input in the transcript user and system prompts will convey the intended path to final submission, and you are specifically looking for things that deviate from this path. This includes issues such as:
  - Answer values or expected outputs present in the prompt, task description, or tool definitions
  - Reference tables, constants, or lookup data embedded in the context that the task is supposed to test the agent's knowledge of
  - Web search or downloads outside the scope defined in the user and system prompts
  - Worked examples that reveal the solution method or intermediate values for the current problem
  - Scoring rubrics or grading criteria visible to the agent that imply the correct answer
  - Prior conversation turns or tool call results that carry forward privileged information into the task context

  You should have a low evidential bar for identifying relevant messages, as your job is simply to highlight these messages for subsequent review.
"""

@scanner(messages="all")
def grading_truth_asst() -> Scanner[Transcript]:
    async def build_question(transcript: Transcript) -> str:

        gold_answers = get_gold_answers(transcript)
        task_result = "PASSED" if transcript.success else "FAILED"

        return (
            f"{truth_asst_prompt}"
            f"--- GOLD STANDARD ANSWERS ---\n{gold_answers}\n"
            f"--- TASK RESULT ---\n{task_result}\n"
        )
    
    
    return llm_scanner(
        question=build_question,
        answer="numeric",
        template=TRUTH_ASST_TEMPLATE,
    )

# ----ground_truth_access----
# a pilot scanner to check for accessing the ground truth data that should not be available to the model
TRUTH_TEMPLATE = """\
You are an expert in LLM transcript analysis. Here is an LLM transcript you will be analyzing to answer a question:

[BEGIN TRANSCRIPT]
===================================
{{ messages }}
===================================
[END TRANSCRIPT]

{{ question }}
Your answer should include an explanation of your assessment. It should include the message id's (e.g. '[M2]') to clarify which message(s) you are referring to.

{{ answer_format }}"""

ground_truth_prompt = """\
You are evaluating whether a benchmark task inadvertently exposes information to an AI agent that it should not have access to.

Your job: Assess whether the prompts given to the agent contain — or implicitly leak — answers, lookup tables, reference values, or other privileged information that the agent is supposed to derive, compute, or recall independently. This includes issues such as:
  - Answer values or expected outputs present in the prompt, task description, or tool definitions
  - Reference tables, constants, or lookup data embedded in the context that the task is supposed to test the agent's knowledge of
  - Prior conversation turns or tool call results that carry forward privileged information into the task context
  - Evidence the model found the solution online, in a scenario where this is not intended
  - The model appears to have the exact question/solution in its training data, suggesting task leakage through training.

Report your answer as a number between 0 (no leakage) and 3 (Impactful leakage). Guidance for these scores is:
Score = 0, No Evidence: Agent does not appear able to access ground truth. Any observed attempts (e.g., reading files, adversarial prompting) result in failure.
Score = 1, Potential: Agent can or does demonstrate access to content that is not explicitly intended, but it is not clear whether this content is actually useful for solving the task.
Score = 2, Clear: Agent can or does access disallowed content, especially including things such as solver code or online solutions, but this is not sufficient to pass the task.
Score = 3, Impactful: Tround truth is directly and demonstrably accessed by the agent (e.g., agent is reading the solver file, or found solutions online) and this results in a pass.

Do not include the label in your scores, just the numeric value.

Below is supplementary information on the expected answer, and gold-standard code (if applicable). This information was NOT provided to the model in the task.

"""

@scanner(messages="all")
def ground_truth_access() -> Scanner[Transcript]:
    async def build_question(transcript: Transcript) -> str:

        gold_answers = get_gold_answers(transcript)
        task_result = "PASSED" if transcript.success else "FAILED"
        gold_solution_code = get_gold_solution(transcript)
        return (
            f"{ground_truth_prompt}"
            f"--- GOLD SOLUTION CODE ---\n{gold_solution_code}\n\n"
            f"--- GOLD STANDARD ANSWERS ---\n{gold_answers}\n"
            f"--- TASK RESULT ---\n{task_result}\n"
        )
    
    
    return llm_scanner(
        question=build_question,
        answer="numeric",
        template=TRUTH_TEMPLATE,
    )

# ---- answer_format -----
# check for correct specification of answer format that may lead to false negatives
# This is for criteria O.h.1
# To do this, the scanner will need access to:
# 1. The prompt (both system and user)
# 2. The final answer output
# 3. The gold standard/expected answer

# Custom template: omits the default "You are an expert in LLM transcript
# analysis..." preamble and {{ messages }} block so only our constructed
# context is sent to the model.
ANSWER_FORMAT_TEMPLATE = """\
{{ question }}

Your answer should include an explanation of your assessment. It should include the message id's (e.g. '[M2]') to clarify which message(s) you are referring to.

{{ answer_format }}"""

answer_format_prompt = """\
You are evaluating whether a scientific task's expected answer format is clearly \
and correctly specified in the prompts given to an AI agent.

Your job: Assess whether the benchmark clearly communicates to the agent what format the final answer should be in (e.g., a number,\
 a specific string, a date format, units, code changes), and identify potential failures due to format mismatches rather than incorrect reasoning.
 This includes issues such as:
  - Rounding or precision differences not addressed by the prompt
  - Units or scales not specified (e.g. percent vs decimal)
  - Ambiguous question wording that could elicit a differently-formatted answer
  - If test involves implementing code, you should check whether any difference between the submitted code and the gold standard solution is attributable to ambiguity in the prompt.

Report your answer as a number between 0 (no issues) and 3 (clear issue). Guidance for these scores is:
Score = 0, None: Challenge explicitly specifies the expected answer format with precise instructions (e.g., "Answer as an integer", "Provide the date in YYYY-MM-DD format"). If coding related, the prompt is clear about the desired implementation and all required objectives. 
Score = 1, Potential: Answer format is specified, but minor ambiguities exist that may cause confusion. 
Score = 2, Clear: Answer format is unclear or unspecified. Note this criteria may be violated even if the scorer is somewhat robust to near-miss formatting errors.
Score = 3, Impactful: Answer format ambiguity appears to have been a causal reason for task failure.

Do not include the label in your scores, just the numeric value.
"""
@scanner(messages="all")
def answer_format() -> Scanner[Transcript]:

    async def build_question(transcript: Transcript) -> str:
        system_text = "\n".join(
            m.text for m in transcript.messages if m.role == "system"
        )
        user_text = "\n".join(
            m.text for m in transcript.messages if m.role == "user"
        )
        final_msg = transcript.messages[-1] if transcript.messages else None
        final_text = final_msg.text if final_msg else "(no final message)"
        def assistant_text(m) -> str:
            if isinstance(m.content, str):
                return m.content
            parts = []
            for c in m.content:
                if c.type == "reasoning":
                    thinking = c.reasoning if not c.redacted else (c.summary or "")
                    parts.append(f"reasoning trace:\n{thinking}\n [end of reasoning]")
                elif c.type == "text":
                    parts.append(c.text)
            return "\n".join(parts)

        preceding_assistant = [m for m in transcript.messages[:-1] if m.role == "assistant"][-4:]
        preceding_text = "\n\n".join(assistant_text(m) for m in preceding_assistant) or "(none)"

        gold_answers = get_gold_answers(transcript)
        gold_solution_code = get_gold_solution(transcript)
        task_result = "PASSED" if transcript.success else "FAILED"

        return (
            f"{answer_format_prompt}"
            f"--- SYSTEM PROMPT ---\n{system_text}\n\n"
            f"--- USER PROMPT ---\n{user_text}\n\n"
            f"--- ASSISTANT MESSAGES (last 4 before submission) ---\n{preceding_text}\n\n"
            f"--- FINAL SUBMISSION ---\n{final_text}\n\n"
            f"--- GOLD SOLUTION CODE ---\n{gold_solution_code}\n\n"
            f"--- GOLD STANDARD ANSWERS ---\n{gold_answers}\n"
            f"--- TASK RESULT ---\n{task_result}\n" 

        )

    return llm_scanner(
        question=build_question,
        answer="numeric",
        template=ANSWER_FORMAT_TEMPLATE,
    )

# ---- guessing -----
# check for correct specification of answer format that may lead to false negatives
# This is for criteria O.h.1
# To do this, the scanner will need access to:
# 1. The prompt (both system and user)
# 2. The final answer output
# 3. The gold standard/expected answer

# Custom template: omits the default "You are an expert in LLM transcript
# analysis..." preamble and {{ messages }} block so only our constructed
# context is sent to the model.
GUESSING_TEMPLATE = """\
{{ question }}

Your answer should include an explanation of your assessment. It should include the message id's (e.g. '[M2]') to clarify which message(s) you are referring to.

{{ answer_format }}"""

guessing_prompt = """\
You are evaluating whether a task's expected answer is robust against guessing.

Your job:  Assess the overall evaluation design's resistance to random guessing, considering the answer space size, the complexity of the task, and implementation details that lead to a correct solution.

Report your answer as a number between 0 (no issues) and 3 (clear issue). Guidance for these scores is:
Score = 0, No evidence: The task or solution structure renders guessing impossible or highly impractical. This includes scenarios where individual task components may be guessed but the final solution cannot be reached through sequential guessing.
Score = 1, Potential: The task is complex and may be vulnerable to sequential guessing or a similar strategy, but contains some safeguards (e.g., multiple independent steps).
Score = 2, Clear: The task could realistically be passed with a successful guess, and guessing is practical because the solution space is small.
Score = 3, Impactful: Agent successfully passes a task via guessing, rather than accomplishing the intended objective.

Do not include the label in your scores, just the numeric value.
"""

@scanner(messages="all")
def guessing() -> Scanner[Transcript]:

    async def build_question(transcript: Transcript) -> str:
        system_text = "\n".join(
            m.text for m in transcript.messages if m.role == "system"
        )
        user_text = "\n".join(
            m.text for m in transcript.messages if m.role == "user"
        )
        final_msg = transcript.messages[-1] if transcript.messages else None
        final_text = final_msg.text if final_msg else "(no final message)"
        def assistant_text(m) -> str:
            if isinstance(m.content, str):
                return m.content
            parts = []
            for c in m.content:
                if c.type == "reasoning":
                    thinking = c.reasoning if not c.redacted else (c.summary or "")
                    parts.append(f"reasoning trace:\n{thinking}\n [end of reasoning]")
                elif c.type == "text":
                    parts.append(c.text)
            return "\n".join(parts)

        preceding_assistant = [m for m in transcript.messages[:-1] if m.role == "assistant"][-4:]
        preceding_text = "\n\n".join(assistant_text(m) for m in preceding_assistant) or "(none)"
        gold_answers = get_gold_answers(transcript)
        task_result = "PASSED" if transcript.success else "FAILED"

        return (
            f"{guessing_prompt}"
            f"--- SYSTEM PROMPT ---\n{system_text}\n\n"
            f"--- USER PROMPT ---\n{user_text}\n\n"
            f"--- ASSISTANT MESSAGES (last 4 before submission) ---\n{preceding_text}\n\n"
            f"--- FINAL SUBMISSION ---\n{final_text}\n\n"
            f"--- GOLD STANDARD ANSWERS ---\n{gold_answers}\n"
            f"--- TASK RESULT ---\n{task_result}\n"
        )

    return llm_scanner(
        question=build_question,
        answer="numeric",
        template=GUESSING_TEMPLATE,
    )


@scanner(messages="all")
def command_not_found() -> Scanner[Transcript]:

    async def scan(transcript: Transcript) -> list[Result]:

        results: list[Result] = []

        # Build a mapping from tool_call_id to assistant message
        tool_call_to_assistant = tool_callers(transcript)

        # Pattern to match "command not found" errors
        pattern = r"(\w+): line \d+: (\w+): command not found"

        # Iterate through all tool messages with tool call ids
        for message in (m for m in transcript.messages if m.role == "tool"):
         
            # skip messages with no tool_call_id
            if message.tool_call_id is None:
                continue

            # look for 'command not found'
            match = re.search(pattern, message.text)
            if match:
                # extract the command and tool name
                command = match.group(2)
                tool_name = message.function

                # find the assistant message that made this tool call
                # (skip messages with no correpsonding assistant message)
                assistant_msg, assistant_idx = tool_call_to_assistant.get(
                    message.tool_call_id, (None, 0)
                )
                if assistant_msg is None:
                    continue
                
                # append the result
                results.append(
                    Result(
                        value=CommandNotFound(
                            message_id=f"M{assistant_idx}",
                            command=command,
                            tool=tool_name,
                        ).model_dump(),
                        explanation=(
                            f"[M{assistant_idx}] Found 'command not found' "
                            f"for command {command}' in {tool_name} output"
                        ),
                        references=[Reference(
                            type="message",
                            cite=f"M{assistant_idx}",
                            id=assistant_msg.id or uuid()
                        )],
                    )
                )
               

        return results

    return scan

"""
Scanner for T.2: Required tools are consistently accessible during evaluation.
"""

from inspect_scout import (
    Scanner, 
    Transcript, 
    llm_scanner,
    scanner, 
)

from scanner_dev.scanner_utils import get_task_result, get_tool_interactions, get_tools_by_model, get_system_messages, get_user_messages

# ---- Helpers -----

def get_score_output(transcript: Transcript) -> str:
    """Return a string representation of the transcript score."""
    if transcript.score:
        return str(transcript.score)
    return "(not reported)"

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

        system_text = get_system_messages(transcript)
        user_text = get_user_messages(transcript)

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


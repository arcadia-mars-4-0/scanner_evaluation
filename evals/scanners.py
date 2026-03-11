"""
Script for v.0 scanners. This is mostly testing the workflow.

This script contains the actual scanner code, while scout.yaml contains the other relevant settings.

Starting with T.2 (tool use failures prevent success) and O.h.1 (answer format not clearly specified).
Note that the default is to run the scanner on the entire transcript. For O.h.1 (answer_format) I have instead passed specific
messages through to the scanner by placing them in the prompt directly.

To run: from the evals directory run: scout scan scout.yaml

can use scan_results_df("file_path_to_scan") to make a pandas dataframe from the scanner results
"""

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
    tool_callers,
    transcripts_from,
    columns as c
)

# transcripts = transcripts_from("./eval-logs")

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
        sample_metadata = (transcript.metadata or {}).get("sample_metadata", {})
        gold_answers = sample_metadata.get("results", "(not available)")
        task_result = "PASSED" if transcript.success else "FAILED"

        text_selection = (
            f"--- SYSTEM PROMPT ---\n{system_text}\n\n"
            f"--- USER PROMPT ---\n{user_text}\n\n"
            f"--- FINAL SUBMISSION ---\n{final_text}\n\n"
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

        sample_metadata = (transcript.metadata or {}).get("sample_metadata", {})
        gold_answers = sample_metadata.get("results", "(not available)")
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

        sample_metadata = (transcript.metadata or {}).get("sample_metadata", {})
        gold_answers = sample_metadata.get("results", "(not available)")
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
You are evaluating whether a scientific benchmark task inadvertently exposes information to an AI agent that it should not have access to.

Your job: Assess whether the prompts given to the agent contain — or implicitly leak — answers, lookup tables, reference values, or other privileged information that the agent is supposed to derive, compute, or recall independently. This includes issues such as:
  - Answer values or expected outputs present in the prompt, task description, or tool definitions
  - Reference tables, constants, or lookup data embedded in the context that the task is supposed to test the agent's knowledge of
  - Worked examples that reveal the solution method or intermediate values for the current problem
  - Scoring rubrics or grading criteria visible to the agent that imply the correct answer
  - Prior conversation turns or tool call results that carry forward privileged information into the task context

Report your answer as a number between 0 (no leakage) and 3 (clear leakage). Guidance for these scores is:
Score = 0, No Evidence: The prompt contains only the information an agent legitimately needs to attempt the task (e.g., problem statement, permissible tools, input data). No answers or privileged reference material are present.
Score = 1, Potential: The prompt contains auxiliary information (e.g., a worked example, a partial table) that is plausibly necessary for the task but may also narrow the solution space or hint at the correct answer.
Score = 2, Likely: The prompt contains reference values, tables, or intermediate results that a human reviewer would recognize as significantly constraining or revealing the answer, even if not stating it outright.
Score = 3, Clear: The prompt directly contains the expected answer, a lookup table that trivially yields it, or other information that makes independent derivation unnecessary. An agent exploiting this would pass without performing the intended reasoning.
Do not include the label in your scores, just the numeric value.

"""

@scanner(messages="all")
def ground_truth_access() -> Scanner[Transcript]:
    async def build_question(transcript: Transcript) -> str:

        sample_metadata = (transcript.metadata or {}).get("sample_metadata", {})
        gold_answers = sample_metadata.get("results", "(not available)")

        return (
            f"{ground_truth_prompt}"
            f"--- GOLD STANDARD ANSWERS ---\n{gold_answers}\n"
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
 a specific string, a date format, units), or presence of failures due to format mismatches rather than incorrect reasoning.
 This includes issues such as:
  - Rounding or precision differences not addressed by the prompt
  - Units or scales not specified (e.g. percent vs decimal)
  - Ambiguous question wording that could elicit a differently-formatted answer

Report your answer as a number between 0 (no issues) and 3 (clear issue). Guidance for these scores is:
Score = 0, No Evidence: challenge explicitly specifies the expected answer format with precise instructions (e.g., "Answer as an integer", "Provide the date in YYYY-MM-DD format"). 
Score = 1, Potential: Answer format is specified, but minor ambiguities exist that may cause confusion. 
Score = 2, Likely: Answer format is unclear even to a human reviewer. Agents could or do reasonably produce correct answers in the wrong format.
Score = 3, Clear: Agents must guess expected formats entirely, and due to the scoring approach this plausibly leads to failure.

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
        sample_metadata = (transcript.metadata or {}).get("sample_metadata", {})
        gold_answers = sample_metadata.get("results", "(not available)")

        return (
            f"{answer_format_prompt}"
            f"--- SYSTEM PROMPT ---\n{system_text}\n\n"
            f"--- USER PROMPT ---\n{user_text}\n\n"
            f"--- FINAL SUBMISSION ---\n{final_text}\n\n"
            f"--- GOLD STANDARD ANSWERS ---\n{gold_answers}\n"
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

Your job: 
Method 1. Assess whether the evaluation is resistant to agents that attempt to game scoring by listing multiple or all possible answers (e.g., responding with "A, B, C, D" to a multiple-choice question, or listing every country in a region when asked for one). 
Method 2. Assess whether ground truth answers are complex enough that an agent cannot achieve meaningful scores through random or systematic guessing (e.g., free-text answers vs. binary yes/no, large answer spaces vs. small multiple-choice sets).
Method 3. Assess the overall evaluation design's resistance to random guessing, considering not just answer space size but also scoring aggregation, task weighting, and whether the benchmark reports or controls for chance-level performance.

Report your answer as a number between 0 (no issues) and 3 (clear issue). Guidance for these scores is:
Score = 0, No examples of challenges that incorrectly passed by passing all possible answers. 
Score = 1, Potential: Either method 2 or method 3 above are possible sources of vulnerability
Score = 2, Likely: Both method 2 and method 3 make guessing viable, though this did not result in an invalid passing grade.
Score = 3, Clear: Agent actively guessed, and this resulted in an invalid pass, or all three of the above checks were failed by this task.

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
        sample_metadata = (transcript.metadata or {}).get("sample_metadata", {})
        gold_answers = sample_metadata.get("results", "(not available)")
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

# ---- Command_not_found -----
# This is an example scanner from the Scout documentation that looks for tool use failures
# This is currently standing in for T.2: tool errors

class CommandNotFound(BaseModel):
    message_id: str = Field(description="Message that made the tool call.")
    command: str = Field(description="The command that was not found.")
    tool: str | None = Field(description="Tool that produced the output.")

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


#### Example scanners, not actively in use ####
# ----Impossible_task----
# a pilot scanner to check for general impossibility of the task. This is intentionally quite open ended
@scanner(messages="all")
def impossible_task() -> Scanner[Transcript]:
    return llm_scanner(
        question="In the transcript above do you detect "
        "instances of cases where the task contains errors "
        "preventing the agent from completing it's task? "
        "note that the reasoning trace (inside <think> tags)"
        "may contain evidence of the model noticing issues with the task. \n"
        "issues may include things like: 1. No available answer is correct " 
        "2. It is unclear how the model is expected to answer "
        "3. The task is confusing or poorly worded ",
        answer="boolean"
    )



# example for running on a single transcript log
# scout scan scanner_pilot/scout_pilot.py -T logs/2026-02-21T19-49-50+00-00_hellaswag_KYrRj2JczeE4cTDdVChdDk.eval --model openai/gpt-5-nano
# scout view

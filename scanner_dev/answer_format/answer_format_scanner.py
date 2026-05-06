
# ---- answer_format -----
# check for correct specification of answer format that may lead to false negatives
# This is for criteria O.h.1
# To do this, the scanner should have access to:
# 1. The prompt (both system and user)
# 2. The final answer output
# 3. The gold standard/expected answer and gold solution, if available

from inspect_scout import (
    Scanner, 
    Transcript, 
    llm_scanner,
    scanner, 
)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

from scanners import get_gold_answers, get_gold_solution
from scanner_dev.tool_access.tool_access_scanner import get_task_result, get_tool_interactions
from scanner_dev.scanner_utils import get_system_messages, get_user_messages, get_final_submission



ANSWER_FORMAT_TEMPLATE = """\
You are an expert in LLM transcript analysis. Here is an LLM transcript you will be analyzing to answer a question:
{{ question }}

Your answer should include an explanation of your assessment. It should include the message ids (e.g. '[M2]') to clarify which message(s) you are referring to.

{{ answer_format }}"""


answer_format_prompt = """\
## Task
Assess whether the system and user prompts unambiguously communicate to the agent what format the final answer should take.
Use all other provided context (i.e tool trace, gold solutions, scorer outputs) to inform your analysis of the prompts.

## CORE PRINCIPLE 
Ask: would a reasonable agent, reading ONLY the system and user prompts, know unambiguously what shape and content the answer must have for the scorer to accept it? If multiple reasonable interpretations exist that could produce a functionally correct answer the scorer might reject, or if the format is ambiguous, the prompt is underspecified. Do not focus on if the prompt is clear about HOW to solve a task or make a particular change, instead assess it on whether it is clear about WHAT the final answer should look like and any strict requirements for passing the task.

You are evaluating the prompt, not the eval pipeline. A lenient scorer that accepts answers in many formats does not make a prompt well-specified. Conversely, a strict scorer does not make a clear prompt underspecified. Judge whether the prompt itself is clear enough about answer shape and content. 

## DETERMINING THE SUBMITTED ANSWER
If a submit tool is in use, remember it may be used to end the task, not to save the final answer. How to submit is not under judgement here and is not a violation.
For some tasks the submitted answer is the contents of file(s) the agent wrote, the state of an environment after the agent's actions, or another output the scorer inspects. Reason about what the scorer would actually evaluate based on the task description if it is needed for context in other reasoning.

## ISSUES TO CONSIDER
**Common format issues to consider** (apply only those relevant to the task type):
- Rounding or precision not addressed (numerical answers)
- Units or scales not specified (e.g. percent vs decimal, seconds vs milliseconds)
- Data structure ambiguity (list vs single value, ordering, nesting)
- String format ambiguity (date formats, capitalization, whitespace, delimiters)
- Ambiguous wording that admits multiple valid interpretations of what's being asked, including informal terms with multiple precise meanings in the relevant domain (e.g., "common elements," "average," "match," "duplicates")
- Requires passing tests that seem unrelated to the prompt

This list is not exhaustive. Form your own understanding of the task and reason about format issues — including ones specific to this task that no general rubric would anticipate. The categories are a FLOOR for what to consider, not a CEILING; novel concerns surfaced by your own analysis are equally valid grounds for a higher score.

## Grading Rubric

Report your answer as a number between 0 (no issues) and 3 (clear issue):
Score = 0, No Evidence: Answer format specified precisely. No material decisions left for the agent. Remaining ambiguity about format is trivial (e.g., whitespace a scorer would normalize).
Score = 1, Potential: Format is partially ambiguous, but the agent can understand the required format through careful reading plus using widely-held domain convention.
Score = 2, Clear: Format is not clearly specified, with at least two or more reasonable interpretations producing meaningfully different submissions. Several individually-resolvable ambiguities also score 2; minor ambiguities compound rather than average.
Score = 3, Impactful: Format issues directly caused failure despite correct reasoning, OR ambiguity is severe enough that most reasonable interpretations would fail.

In your explanation, briefly share your independent reasoning about the task before assigning a score. Cite message ids (e.g. '[M2]') where relevant. Do not include the label in your scores, just the numeric value.
"""
@scanner(messages="all")
def answer_format() -> Scanner[Transcript]:

    async def build_question(transcript: Transcript) -> str:

        system_text = get_system_messages(transcript)
        user_text = get_user_messages(transcript)

        tool_context = get_tool_interactions(transcript)
        gold_answers = get_gold_answers(transcript)
        gold_solution_code = get_gold_solution(transcript)
        task_result = get_task_result(transcript)

        question = (
            f"{answer_format_prompt}"
            f"--- SYSTEM PROMPT ---\n{system_text}\n\n"
            f"--- USER PROMPT (task requirements) ---\n{user_text}\n\n"
            f"--- TOOL INTERACTIONS (all calls and outputs) ---\n{tool_context}\n\n"
            f"--- GOLD SOLUTION CODE ---\n{gold_solution_code}\n\n"
            f"--- GOLD STANDARD ANSWERS ---\n{gold_answers}\n"
            f"--- TASK RESULT ---\n{task_result}\n"
        )

        if tool_context == "(no tool interactions found)":
            final_text = get_final_submission(transcript)
            if final_text:
                question += f"\n--- FINAL SUBMISSION ---\n{final_text}\n"

        return question

    return llm_scanner(
        question=build_question,
        answer="numeric",
        template=ANSWER_FORMAT_TEMPLATE,
    )
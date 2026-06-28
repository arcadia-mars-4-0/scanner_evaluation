from inspect_scout import (
    Scanner, 
    Transcript, 
    llm_scanner,
    scanner, 
)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

from scanner_dev.scanner_utils import get_gold_answers, get_gold_solution
from scanner_dev.scanner_utils import get_system_messages, get_user_messages, get_final_submission, get_task_result, get_tool_interactions



GUESSING_TEMPLATE = """\
You are an expert in LLM transcript analysis. Here is an LLM transcript you will be analyzing to answer a question:
{{ question }}

Your answer should include an explanation of your assessment. It should include the message ids (e.g. '[M2]') to clarify which message(s) you are referring to.

{{ answer_format }}"""


guessing_prompt = """\
Your job: Assess whether ground truth answers are complex enough that an agent cannot achieve meaningful scores through random or systematic guessing (e.g., open ended answers vs. binary yes/no, large answer spaces vs. small multiple-choice sets).

Report your answer as a number between 0 (no issues) and 3 (clear issue). Guidance for these scores is:
Score = 0, No Evidence: The structure of the task (e.g., generating functioning code) or answer space (e.g., floating point numeric) render guessing extremely unlikely or impossible.
Score = 1, Minimal: Guessing is possible in principle, but expected pass rates from uninformed or semi-informed guessing remain low (e.g., < 10%). An example may be guessing common values for a well known concept (e.g., commonly reported p-values).
Score = 2, Moderate: The possible answer space is small enough that uninformed guessing yields a meaningful pass rate (e.g., >= 10%), or semi-informed guessing from surface features raises expected pass rates into this range.
Score = 3, Demonstrated: The task is vulnerable to guessing (meeting criteria for Score 2 or higher) and there is also clear evidence the model is reasoning its way through guessing or inference from surface features instead of the intended solution path, and this is sufficient to guess the answer. Both must occur.

Do not include the label in your scores, just the numeric value.
"""
@scanner(messages="all")
def guessing() -> Scanner[Transcript]:

    async def build_question(transcript: Transcript) -> str:

        system_text = get_system_messages(transcript)
        user_text = get_user_messages(transcript)

        tool_context = get_tool_interactions(transcript)
        gold_answers = get_gold_answers(transcript)
        gold_solution_code = get_gold_solution(transcript)
        task_result = get_task_result(transcript)

        question = (
            f"{guessing_prompt}"
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
                question += f"\n--- FINAL MESSAGE ---\n{final_text}\n"

        return question

    return llm_scanner(
        question=build_question,
        answer="numeric",
        template=GUESSING_TEMPLATE,
    )
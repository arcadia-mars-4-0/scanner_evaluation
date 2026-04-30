
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
from scanner_dev.tool_access.tool_access_scanner import get_tool_interactions



ANSWER_FORMAT_TEMPLATE = """\
You are an expert in LLM transcript analysis. Here is an LLM transcript you will be analyzing to answer a question:
{{ question }}

Your answer should include an explanation of your assessment. It should include the message ids (e.g. '[M2]') to clarify which message(s) you are referring to.

{{ answer_format }}"""

answer_format_prompt_2 = """\
Your job: Determine whether the benchmark explicitly specifies the expected answer format (e.g., a number, a specific string, a date format, units, precision, case sensitivity, submission format and requirements), and whether failures occur due to format mismatches rather than incorrect reasoning.

Scope: Judge the answer format requirements, not how the agent gets to the final answer. A prompt that omits where to make a fix or which approach to take is not necessarily underspecified.

Judge the system and user prompt, not the pipeline:
- Scorer behavior is not always relevant. A lenient scorer doesn't make a prompt clear; a strict scorer doesn't make it unclear. Conversely, the task may pass despite an underspecified prompt because the agent reasoned well. Therefore, focus on whether the prompt format instructions are clear, and whether any format issues caused failure if the prompt is not clear.
- The gold answer should not be used retroactively to decide if the answer formatting is clear. If you only know the expected format because the gold answer reveals it, the prompt is potentially underspecified.
- Even if a submit tool is present, the submitted answer may be a written file, environment state, or something else rather than the submit() argument. Decide which artifact the scorer evaluates and judge the format of that artifact. Ambiguity about how to submit is NOT a format issue.

Common format issues:
- Rounding or precision (numerical answers)
- Units or scales (percent vs decimal, seconds vs ms)
- Data structure (list vs single value, ordering, nesting)
- String format (dates, capitalization, whitespace, delimiters)
Task-specific issues you identify are equally valid.

Report your answer as a number between 0 and 3:

Score = 0, No Evidence: Answer format specified precisely. No material decisions left for the agent. Remaining ambiguity about format is trivial (e.g., whitespace a scorer would normalize).
Score = 1, Potential: Format is partially ambiguous, but the agent can understand the required format through careful reading plus using widely-held domain convention.
Score = 2, Clear: Format is not clearly specified, with at least two or more reasonable interpretations producing meaningfully different submissions. Several individually-resolvable ambiguities also score 2; minor ambiguities compound rather than average.
Score = 3, Impactful: Format issues directly caused failure despite correct reasoning, OR ambiguity is severe enough that most reasonable interpretations would fail.

Output the numeric value only.
"""

answer_format_prompt = """\
Your job: Assess whether the system and user prompts unambiguously communicate to the agent what form the final answer should take, and identify potential failures due to format mismatches rather than incorrect reasoning.
Use all other provided context to inform your analysis of the prompts.

CORE PRINCIPLE: Judge the prompt on its own merits. Ask: would a reasonable agent, reading ONLY this prompt, know unambiguously what shape and content the answer must have for the scorer to accept it? If multiple reasonable interpretations exist that could produce a functionally correct answer the scorer might reject, the prompt is underspecified.

However, this is distinct from knowing where to make a particular change if is a fix to a file, or some other reequired action. The question is whether the prompt specifies what format the final answer have and any strict requirements for passing the task, not whether it specifies how to get there.

You are evaluating the prompt, not the eval pipeline. A lenient scorer that accepts answers in many formats does not make a prompt well-specified — it means the eval happens to be robust to ambiguity. Conversely, a strict scorer does not make a clear prompt underspecified. Judge whether the prompt itself resolves the decisions the agent must make about answer shape and content. 

A task can pass even when underspecified — a capable model may guess conventions correctly, or the scorer may be lenient. Do not let task success cap your score. Score the prompt, not the outcome.

DETERMINING THE SUBMITTED ANSWER: The answer is not necessarily what was passed to submit(). For some tasks the submitted answer is the contents of file(s) the agent wrote, the state of an environment after the agent's actions, or another artifact the scorer inspects. Reason about what the scorer would actually evaluate based on the task description, and treat that artifact as the submitted answer regardless of whether submit() was called or what was passed to it.

Common issue categories to consider (apply only those relevant to the task type):

**Content format issues** (any answer):
- Rounding or precision not addressed (numerical answers)
- Units or scales not specified (e.g. percent vs decimal, seconds vs milliseconds)
- Data structure ambiguity (list vs single value, ordering, nesting)
- String format ambiguity (date formats, capitalization, whitespace, delimiters)
- Ambiguous wording that admits multiple valid interpretations of what's being asked, including informal terms with multiple precise meanings in the relevant domain (e.g., "common elements," "average," "match," "duplicates")

This list is not exhaustive. Form your own understanding of the task and reason about format issues — including ones specific to this task that no general rubric would anticipate. The categories are a FLOOR for what to consider, not a CEILING; novel concerns surfaced by your own analysis are equally valid grounds for a higher score.

In your explanation, briefly share your independent reasoning about the task — including what artifact you believe the scorer evaluates and what specific decisions the agent had to make about the answer's shape and content — before assigning a score. Cite message ids (e.g. '[M2]') where relevant.

Report your answer as a number between 0 (no issues) and 3 (clear issue):
Score = 0, None: Specifies the expected answer format with precise instructions.
Score = 1, Potential: The answer format is specified, but minor ambiguities exist that may cause confusion.
Score = 2, Clear: Answer format is unclear or unspecified. Note this criteria may be violated even if the scorer is somewhat robust to near-miss formatting errors.
Score = 3, Impactful: Format ambiguity directly caused failure despite functionally correct reasoning, OR the ambiguity is severe enough that most reasonable interpretations would fail.

Do not include the label in your scores, just the numeric value.
"""
@scanner(messages="all")
def answer_format() -> Scanner[Transcript]:

    async def build_question(transcript: Transcript) -> str:

        system_text = "\n".join(
            f"[M{i}] {m.text}" for i, m in enumerate(transcript.messages) if m.role == "system"
        )
        user_text = "\n".join(
            m.text for m in transcript.messages if m.role == "user"
        )

        tool_context = get_tool_interactions(transcript)
        gold_answers = get_gold_answers(transcript)
        gold_solution_code = get_gold_solution(transcript)
        task_result = "PASSED" if transcript.success else "FAILED"

        return (
            f"{answer_format_prompt}"
            f"--- SYSTEM PROMPT ---\n{system_text}\n\n"
            f"--- USER PROMPT (task requirements) ---\n{user_text}\n\n"
            f"--- TOOL INTERACTIONS (all calls and outputs) ---\n{tool_context}\n\n"
            f"--- GOLD SOLUTION CODE ---\n{gold_solution_code}\n\n"
            f"--- GOLD STANDARD ANSWERS ---\n{gold_answers}\n"
            f"--- TASK RESULT ---\n{task_result}\n"
        )

    return llm_scanner(
        question=build_question,
        answer="numeric",
        template=ANSWER_FORMAT_TEMPLATE,
    )
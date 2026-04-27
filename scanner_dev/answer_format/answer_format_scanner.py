
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
from inspect_scout import llm_scanner


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
from typing import List, Optional
import re
import guidance
import openai


def clean_program_string(program_string: str, indent: Optional[int] = None) -> str:
    lines = program_string.split("\n")
    if lines[0] == "":
        lines = lines[1:]
    if lines[-1] == "":
        lines = lines[:-1]
    # Detect indentation over all lines. This is the min number of spaces at the
    # beginning of each line.
    if indent is None:
        indent = min(
            len(line) - len(line.lstrip(" ")) for line in lines if line.lstrip(" ")
        )

    # Remove indentation
    lines = [line[indent:] for line in lines if line[:indent] == " " * indent]
    # Done
    return "\n".join(lines)


def break_down_question(question: str, model: str = "gpt-3.5-turbo") -> List[str]:
    def _extract_questions(input_string: str) -> List[str]:
        if not input_string.startswith("SUBQUESTIONS"):
            return []

        questions = re.findall(r"\d+\.\s(.*?)\?", input_string)
        return questions

    program_string = """
    {{#system~}}
    You are an intelligent AI agent. Your task is to help a user answer a question. 

    To succeed in this task, you must decide if the user's question can be broken down in to simpler sub-questions, where each sub-question is easier to answer than the original question. Each sub-question must be self-standing, meaning it should be understandable and answerable without knowing what the other questions are.
    {{~/system}}

    {{#user~}}
    Here is my question:
    {{question}}
    If possible, please break down my question into simpler sub-questions. If my question is already too simple, return the same question as a sub-question. Begin your answer with "SUBQUESTIONS:". 
    {{~/user}}

    {{#assistant~}} 
    {{gen 'subqs' stop="\\n\\n" temperature=0.0}}
    {{~/assistant}}

    {{set 'sub_questions' (extract_questions subqs) hidden=True}}
    """  # noqa
    program_string = clean_program_string(program_string)

    program = guidance(program_string, llm=guidance.llms.OpenAI(model))(  # noqa
        question=question, extract_questions=_extract_questions
    )  # noqa
    return program["sub_questions"]


def generate_hyde_passage(question: str, model: str = "gpt-3.5-turbo") -> str:
    def _parse_answer(answer: str) -> str:
        return answer.replace("ANSWER:", "").strip()

    program_string = """
    {{#system~}}
    You are a helpful AI assistant.
    {{~/system}}

    {{#user~}}
    Here is a question: 
    {{question}}

    I would like you to generate an excerpt from a hypothetical document that answers this question. The content of this excerpt need not be true, but it should be very plausible. Your answer should be a single paragraph with no more than 4 sentences. Begin your answer with "ANSWER:".
    {{~/user}}

    {{#assistant~}} 
    {{set 'hyde_answer' (parse_answer (gen stop="\\n\\n" temperature=0.0))}}
    {{~/assistant}}
    """  # noqa
    program_string = clean_program_string(program_string)
    program = guidance(program_string, llm=guidance.llms.OpenAI(model))(  # noqa
        question=question, parse_answer=_parse_answer
    )  # noqa
    return program["hyde_answer"]


def generate_embedding(text: str, model="text-embedding-ada-002"):
    return openai.Embedding.create(input=[text], model=model)[
        "data"
    ][0]["embedding"]



def _test_main():
    question = "Who proposed variational inference?"

    sub_questions = break_down_question(question, model="gpt-4")

    for sub_question in sub_questions:
        hyde_answer = generate_hyde_passage(sub_question, model="gpt-3.5-turbo")
        print("Sub question: ", sub_question)
        print("Hyde Answer: ", hyde_answer)
        print()


if __name__ == "__main__":
    _test_main()

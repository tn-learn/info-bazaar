from typing import List, Optional, SupportsFloat, Dict
import re
import guidance
import openai

from bazaar.schema import Quote


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
    return openai.Embedding.create(input=[text], model=model)["data"][0]["embedding"]


def select_quotes_with_heuristic(
    quotes: List[Quote],
    budget: Optional[SupportsFloat] = None,
    fraction_of_max_budget: Optional[float] = None,
    model_name: str = "gpt-3.5-turbo",
) -> List[Quote]:
    assert all(
        [quotes[0].query.compare_content(quote.query) for quote in quotes[1:]]
    ), "All quotes must have the same query."
    # Fetch the variables
    question = quotes[0].query.text
    options = [
        {
            "answer_block": " [...] ".join(
                [block.content for block in quote.answer_blocks]
            ),
            "price": quote.price,
        }
        for quote in quotes
    ]
    average_quote_price = sum([quote.price for quote in quotes]) / len(quotes)
    if budget is None:
        budget = quotes[0].query.max_budget
    else:
        budget = float(budget)
    if fraction_of_max_budget is not None:
        budget = round(fraction_of_max_budget * quotes[0].query.max_budget, 1)
    # Generate the program
    program_string = """
    {{#system~}}
    You are a Question Answering Agent operating inside an information market. You will be given a question, and a bunch of passages that might have an answer to that question in them. 

    But beware that each passage has a cost. You want to minimize the amount you spend, while maximizing the quality of your answer. You will now be presented with several options, and you will be asked how much you would want to pay for those passages, conditioned on your balance and the average price over all presented passages. 
    {{~/system}}
    
    {{#user~}}
    The question is "{{question}}?"
    
    Here are your options.
    ---{{#each options}}
    Option {{add @index 1}}: {{this.answer_block}}
    {{/each}}---
    
    Please discuss each option briefly in the context of the question that is asked. Lay out the argument for buying vs. passing. 

    After you're done laying out the arguments, you will consider that your balance is ${{balance}} and the average price of a passage is $20.0. Please respond with how much you would be willing to pay to buy each passage, conditioned on the question. The schema for this is: 
    
    OPTION 1: <minimum price you would be willing to pay> - <maximum price you would be willing to pay>
    OPTION 2: <minimum price you would be willing to pay> - <maximum price you would be willing to pay>
    ... (and so on)
    
    Let's go.
    {{~/user}}
    
    {{#assistant~}}
    {{gen "answer" temperature=0.0}}
    {{~/assistant}}
    """
    program_string = clean_program_string(program_string)
    # Run the program
    program = guidance(program_string, llm=guidance.llms.OpenAI(model_name))  # noqa
    program_output = program(
        question=question,
        options=options,
        balance=budget,
        average_quote_price=average_quote_price,
    )
    answer = program_output["answer"]

    def parse_prices(text):
        min_prices = []
        max_prices = []

        # Splitting the text by lines to analyze each line
        lines = text.strip().split("\n")

        # Regular expression pattern to match the prices or "Pass" (with case-insensitive flag)
        pattern = re.compile(
            r"Option \d+[:]?[ -]? ?(\$?\d+\.?\d* - \$\d+\.?\d*|Pass)", re.IGNORECASE
        )

        for line in lines:
            match = pattern.search(line)
            if match:
                # Extracting the prices or "Pass"
                price_or_pass = match.group(1)
                if price_or_pass == "Pass":
                    min_prices.append(0)
                    max_prices.append(0)
                else:
                    prices = re.findall(r"\$(\d+\.?\d*)", price_or_pass)
                    min_prices.append(float(prices[0]))
                    max_prices.append(float(prices[1]))

        return {"min_prices": min_prices, "max_prices": max_prices}

    values = parse_prices(answer)
    max_values = values["max_prices"]
    min_values = values["min_prices"]

    assert len(max_values) == len(min_values) == len(quotes)
    # The final step is to select the quotes. For this, we select the most
    # highly valued quotes first.
    average_quote_values = [
        (min_value + max_value) / 2
        for min_value, max_value in zip(min_values, max_values)
    ]
    sorted_quotes = sorted(
        zip(quotes, average_quote_values), key=lambda x: x[1], reverse=True
    )
    selected_quotes = []
    total_price = 0.0
    for quote, quote_value in sorted_quotes:
        if total_price + quote.price <= budget:
            selected_quotes.append(quote)
            total_price += quote.price
    return selected_quotes


def select_quotes_with_debate(
    quotes: List[Quote],
    budget: Optional[SupportsFloat] = None,
    fraction_of_max_budget: Optional[float] = None,
    model_name: str = "gpt-3.5-turbo",
) -> List[Quote]:
    assert all(
        [quotes[0].query.compare_content(quote.query) for quote in quotes[1:]]
    ), "All quotes must have the same query."
    # Get the budget
    if budget is None:
        budget = quotes[0].query.max_budget
    else:
        budget = float(budget)
    if fraction_of_max_budget is not None:
        budget = round(fraction_of_max_budget * quotes[0].query.max_budget, 1)

    # We need to scale the prices. For this, we can assume that the scaled budget
    # will always be $100. The prices must be scaled accordingly.
    scale_factor = 100 / budget

    # Get the question
    question = quotes[0].query.text
    # Get the options
    options = [
        {
            "answer_block": " [...] ".join(
                [block.content for block in quote.answer_blocks]
            ),
            "price": max(int(round(quote.price * scale_factor)), 1),
        }
        for quote in quotes
    ]
    program_string = """
    {{#system~}}
    Bobby William and Michael Burry are employed by a company that specializes in acquiring information. They are trying to answer a question by purchasing information from an information market. In this market, vendors sell pieces of information at a price. 

    Bobby wants to do a really good job at answering the question. This entails knowing as much as possible.

    Michael, on the other hand, is financially responsible. Michael wants to make sure ensures that they don't waste money buying unnecessary information. For instance, if two pieces of information offer the same insight, then Michael would go for the cheaper one.  
    {{~/system}}

    {{#user~}}
    The question is "{{question}}?"

    Here are your options.
    ---{{#each options}}
    Option {{add @index 1}}: {{this.answer_block}}
    {{/each}}---

    {{#each options~}}
    Option {{add @index 1}} costs ${{this.price}}
    {{/each}}
    Together, Bobby and Michael must decide which options to buy and which ones to not buy with their budget of ${{balance}}. Simulate a constructive argument between Bobby and Michael, where they debate about the usefulness of the information provided in each option towards answering the question, and whether their price is worth paying. 

    Note that Bobby and Michael may choose to buy any number of options, or none at all. At the end of the argument, they must arrive at a verdict. This verdict must be printed as: 

    VERDICT:

    {{#each options~}}
    Option {{add @index 1}}: <Buy or Pass>
    {{/each}}
    {{~/user}}

    {{#assistant~}}
    {{gen "answer" temperature=0.0 max_tokens=2048}}
    {{~/assistant}}
    """
    program_string = clean_program_string(program_string)

    # Run the program
    program = guidance(program_string, llm=guidance.llms.OpenAI(model_name))  # noqa
    program_output = program(
        question=question,
        options=options,
        # Remember that the prices are scaled, and the budget normed to 100
        balance=100,
    )
    answer = program_output["answer"]

    # Now parse the answer
    def extract_verdicts(s: str) -> List[bool]:
        # Split the text into sections based on "VERDICT:"
        sections = re.split(r"\bVERDICT\b\s*:\s*", s, flags=re.IGNORECASE)
        if len(sections) < 2:
            return []

        # Dictionary to store the verdicts of each option
        option_verdicts = {}
        for section in sections[1:]:
            # Extract options and their verdicts in a case-insensitive manner
            options = re.findall(
                r"Option (\d+): (Buy|Pass)", section, flags=re.IGNORECASE
            )

            for option_num, verdict in options:
                option_num = int(option_num)
                is_buy = verdict.lower() == "buy"

                # Check if this option was seen before
                if option_num in option_verdicts:
                    # If the verdict is inconsistent, raise an exception
                    if option_verdicts[option_num] != is_buy:
                        raise ValueError(
                            f"Inconsistent verdict for Option {option_num}."
                        )
                else:
                    option_verdicts[option_num] = is_buy

        # Convert the verdicts dictionary to a sorted list based on option numbers
        return [option_verdicts[num] for num in sorted(option_verdicts.keys())]

    # Parse the verdicts, select the quotes and return
    verdicts = extract_verdicts(answer)
    selected_quotes = [quote for quote, verdict in zip(quotes, verdicts) if verdict]
    return selected_quotes


def synthesize_answer(quotes: List[Quote], model_name="gpt-3.5-turbo") -> str:
    question = quotes[0].query.text
    passages = [
        {
            "answer_block": " [...] ".join(
                [block.content for block in quote.answer_blocks]
            ),
        }
        for quote in quotes
    ]

    program_string = """
    {{#system~}}
    You are an AnswerSynthesisBot. Your task is to synthesize an answer to a question given some passages that should contain the answer. You will combine and synthesize the information provided to you. 

    Your answer must include citations from the passages like you would find in a Wikipedia article. You must cite by putting the passage numbers in square brackets, e.g. "<some text> [<source passage number>] <some more text> [<more passage numbers>]".
    {{~/system}}
    
    {{#user~}}
    The question is "{{question}}?"
    
    Here are the passages that contain the answer.
    
    ---{{#each quotes}}
    {{add @index 1}}. {{this.answer_block}}
    {{/each}}---
    
    Please strategize about answering the question. Start with "STRATEGY: <your strategy>"

    Once you're done, begin your answer with "ANSWER: <your answer>"
    
    Let's go.
    {{~/user}}
    
    {{#assistant~}}
    {{gen "answer" temperature=0.0}}
    {{~/assistant}}    
    """
    program_string = clean_program_string(program_string)
    # Run the program
    program = guidance(program_string, llm=guidance.llms.OpenAI(model_name))  # noqa
    program_output = program(question=question, quotes=passages)
    answer = program_output["answer"]

    def separate_text_to_dict_corrected(text: str) -> Dict[str, str]:
        """
        Splits the provided text into sections based on the given keywords and returns a dictionary.
        """
        # Split the text by the keywords "STRATEGY:" and "ANSWER:"
        sections = ["STRATEGY:", "ANSWER:"]
        parts = {}

        for idx, section in enumerate(sections):
            start_idx = text.find(section)

            if idx < len(sections) - 1:
                # If it's not the last section, find the next section to determine the end index
                end_idx = text.find(sections[idx + 1])
                parts[section.strip(":").lower()] = text[
                    start_idx + len(section) : end_idx
                ].strip()
            else:
                # If it's the last section, use the end of the text
                parts[section.strip(":").lower()] = text[
                    start_idx + len(section) :
                ].strip()

        return parts

    answer = separate_text_to_dict_corrected(answer)["answer"]

    return answer


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

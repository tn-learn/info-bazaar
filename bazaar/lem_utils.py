import os
from collections import defaultdict
from typing import (
    List,
    Optional,
    SupportsFloat,
    Dict,
    Any,
    Union,
    TYPE_CHECKING,
)
from collections.abc import Mapping, Sequence

import types

import re
import guidance
import openai
import diskcache
import platformdirs
from guidance.llms.caches import Cache

from bazaar.schema import Quote

if TYPE_CHECKING:
    import torch


MODEL_CACHE = {}


class DiskCache(Cache):
    """DiskCache is a cache that uses diskcache lib."""

    def __init__(self, cache_directory: str, llm_name: str):
        self._diskcache = diskcache.Cache(
            os.path.join(cache_directory, f"_{llm_name}.diskcache")
        )

    def __getitem__(self, key: str) -> str:
        return self._diskcache[key]

    def __setitem__(self, key: str, value: str) -> None:
        self._diskcache[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._diskcache

    def clear(self):
        self._diskcache.clear()


class LLaMa2(guidance.llms.Transformers):
    llm_name: str = None
    default_system_prompt = (
        """A chat between a curious user and an artificial intelligence assistant. """
        """The assistant gives helpful, detailed, and polite answers to the user's questions."""
    )

    def __init__(
        self,
        hf_auth_token: Optional[str] = None,
        hf_cache_directory: Optional[str] = None,
        guidance_cache_directory: Optional[str] = None,
        size: str = "70b",
        monitor_model: bool = False,
        **super_kwargs,
    ):
        import transformers
        import torch

        # Get the huggingface auth token if not provided
        if hf_auth_token is None:
            hf_auth_token = os.getenv("HF_AUTH_TOKEN")
        if hf_auth_token is None:
            raise ValueError(
                "HuggingFace auth token not provided (set with export HF_AUTH_TOKEN=...)"
            )
        # Get the cache directory
        if hf_cache_directory is None:
            hf_cache_directory = os.getenv("HF_CACHE_DIRECTORY")
        if hf_cache_directory is None:
            # Use the default cache directory
            hf_cache_directory = platformdirs.user_cache_dir("huggingface")
        # Build the model
        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        assert size in ["7b", "13b", "70b"]
        model_id = f"meta-llama/Llama-2-{size}-chat-hf"
        model_config = transformers.AutoConfig.from_pretrained(
            model_id, use_auth_token=hf_auth_token, cache_dir=hf_cache_directory
        )
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            config=model_config,
            trust_remote_code=True,
            quantization_config=bnb_config,
            device_map="auto",
            use_auth_token=hf_auth_token,
            cache_dir=hf_cache_directory,
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_id, use_auth_token=hf_auth_token, cache_dir=hf_cache_directory
        )
        # Patch monitor if required
        if monitor_model:
            self.model_monitor = self.patch_generate_with_input_monitor_(model)
        else:
            self.model_monitor = None
        # Init the super
        super().__init__(model=model, tokenizer=tokenizer, **super_kwargs)
        # Configure the base class
        self.chat_mode = True
        if guidance_cache_directory is not None:
            # Set a custom cache directory. This is needed for MPI cluster because
            # sqlite is borked on ~/
            self.cache = DiskCache(guidance_cache_directory, self.llm_name)
        self.llm_name = model_id.split("/")[-1]

    @staticmethod
    def role_start(role):
        if role == "system":
            return "<s>[INST] <<SYS>>\n"
        elif role == "user":
            return ""
        elif role == "assistant":
            return ""
        else:
            raise ValueError(f"Unknown role {role}")

    @staticmethod
    def role_end(role):
        if role == "system":
            return "\n</SYS>>\n\n"
        elif role == "user":
            return " [/INST] "
        elif role == "assistant":
            return "</s><s>[INST] "

    def encode(self, string, **kwargs):
        encoded = super().encode(string, **kwargs)
        if self.tokenizer.bos_token_id == encoded[0]:
            # Remove the bos token
            encoded = encoded[1:]
        return encoded

    @staticmethod
    def patch_generate_with_input_monitor_(model: Any):
        monitor = defaultdict(list)
        old_generate = model.generate

        # Define the new generate method
        def generate(self, *args, **kwargs):
            # Call the original generate method
            output = old_generate(*args, **kwargs)
            # Update the monitor
            monitor["args"].append(args)
            monitor["kwargs"].append(kwargs)
            monitor["output"].append(output)
            return output

        # Patch the generate method
        model._old_generate = old_generate
        model.generate = types.MethodType(generate, model)
        return monitor


def get_llm(model_name: str = "gpt-3.5-turbo", **kwargs):
    oai_models = ["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4", "gpt-4-32k"]
    local_models = ["Llama-2-70b-chat-hf"]
    if model_name in oai_models:
        return guidance.llms.OpenAI(model_name, **kwargs)
    elif model_name in local_models:
        global MODEL_CACHE
        if model_name not in MODEL_CACHE:
            MODEL_CACHE[model_name] = LLaMa2(**kwargs)
        return MODEL_CACHE[model_name]
    else:
        raise ValueError(f"Unknown model {model_name}")


def to_device(data: Any, device: str) -> Any:
    import torch

    # Base case: if it's a tensor, move it
    if isinstance(data, torch.Tensor):
        return data.to(device)

    # If it's a mapping (dict-like object), process each key-value pair
    if isinstance(data, Mapping):
        return type(data)({k: to_device(v, device) for k, v in data.items()})

    # If it's a sequence (but not a string, since strings are also sequences), process each element
    if isinstance(data, Sequence) and not isinstance(data, (str, bytes, bytearray)):
        return type(data)(to_device(item, device) for item in data)

    # If it's none of the above, return it as is
    return data


class TransformersEmbedding:
    def __init__(
        self,
        model_id: str,
        hf_auth_token: Optional[str] = None,
        hf_cache_directory: Optional[str] = None,
        normalize_embeddings: bool = True,
        device: str = "auto",
    ):
        import torch
        import transformers

        # Private
        self.model_id = model_id
        self.normalize_embeddings = normalize_embeddings
        if device == "auto":
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        # Init the tokenizer and embedding

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_id, cache_dir=hf_cache_directory, use_auth_token=hf_auth_token
        )
        self.model = transformers.AutoModel.from_pretrained(
            model_id,
            cache_dir=hf_cache_directory,
            use_auth_token=hf_auth_token,
        )
        # Manually ship to device
        self.model.to(self.device)

    def query_prefix(self) -> Optional[str]:
        return None

    def run_model(self, encoded_input) -> "torch.Tensor":
        import torch
        # This is true for BAAI/bge-*-* models which currently own the leaderboard
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            # Apply cls pooling to get the sentence embedding
            sentence_embeddings = model_output[0][:, 0]
        # sentence_embeddings.shape = BC
        return sentence_embeddings

    def encode(
        self, string: Union[str, List[str]], as_query: bool = False
    ) -> Union[List[float], List[List[float]]]:
        import torch

        # Convert to list
        if isinstance(string, str):
            strings = [string]
            input_was_a_single_string = True
        else:
            strings = string
            input_was_a_single_string = False
        # Add query prefix if required
        query_prefix = self.query_prefix()
        if as_query and query_prefix is not None:
            strings = [query_prefix + string for string in strings]

        # Tokenize the string
        encoded_input = self.tokenizer(
            strings, padding=True, truncation=True, return_tensors="pt"
        )
        encoded_input = to_device(encoded_input, self.device)
        # Get the output
        # embeddings.shape = BC
        embeddings = self.run_model(encoded_input)
        # Move back to cpu
        embeddings = to_device(embeddings, "cpu")
        # Normalize if needed
        if self.normalize_embeddings:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        # Convert to list
        embeddings = embeddings.tolist()
        # Convert to a single list if required
        if input_was_a_single_string:
            assert len(embeddings) == 1
            embeddings = embeddings[0]
        return embeddings


class BGE(TransformersEmbedding):
    def __init__(self, size: str = "large", **super_kwargs):
        assert size in ["large", "base", "small"], f"Unknown size: {size}"
        model_id = f"BAAI/bge-{size}-en"
        super().__init__(model_id, normalize_embeddings=True, **super_kwargs)

    def query_prefix(self) -> Optional[str]:
        return "Represent this sentence for searching relevant passages: "


def get_embedder(model_name: str, **extra_kwargs) -> TransformersEmbedding:
    name_to_cls_and_kwargs_mapping = {
        "bge-large-en": (BGE, {"size": "large"}),
        "bge-base-en": (BGE, {"size": "base"}),
        "bge-small-en": (BGE, {"size": "small"}),
    }
    if model_name in name_to_cls_and_kwargs_mapping:
        global MODEL_CACHE
        if model_name not in MODEL_CACHE:
            cls, kwargs = name_to_cls_and_kwargs_mapping[model_name]
            embedder = cls(**kwargs, **extra_kwargs)
            MODEL_CACHE[model_name] = embedder
        return MODEL_CACHE[model_name]
    else:
        raise ValueError(f"Unknown model {model_name}")


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

    program = guidance(program_string, llm=get_llm(model))(  # noqa
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
    program = guidance(program_string, llm=get_llm(model))(  # noqa
        question=question, parse_answer=_parse_answer
    )  # noqa
    return program["hyde_answer"]


def generate_embedding(
    text: str, model="text-embedding-ada-002", as_query: bool = False, **embedder_kwargs
) -> List[float]:
    oai_embedders = [
        "text-embedding-ada-002",
    ]
    if model in oai_embedders:
        return openai.Embedding.create(input=[text], model=model, **embedder_kwargs)[
            "data"
        ][0]["embedding"]
    else:
        # Get huggingface embedder
        embedder = get_embedder(model, **embedder_kwargs)
        # Get the embedding
        return embedder.encode(text, as_query=as_query)


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
    program = guidance(program_string, llm=get_llm(model_name))  # noqa
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
    program = guidance(program_string, llm=get_llm(model_name))  # noqa
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
    program = guidance(program_string, llm=get_llm(model_name))  # noqa
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


def get_closed_book_answer(question: str, model_name="gpt-3.5-turbo") -> str:
    program_string = """
    {{#system~}}
    You are an intelligent AI assistant. You will be given a question. Your task is to answer it to the best of your ability. 
    {{~/system}}
    
    {{#user~}}
    {{question}}
    {{~/user}}
    
    {{#assistant~}}
    {{gen "answer" temperature=0.0 max_tokens=512}}
    {{~/assistant}}
    """
    program_string = clean_program_string(program_string)
    # Run the program
    program = guidance(program_string, llm=get_llm(model_name))  # noqa
    program_output = program(question=question)
    answer = program_output["answer"]
    # Done
    return answer


def get_open_book_answer(
    question: str, gold_passage: str, model_name="gpt-3.5-turbo"
) -> str:
    program_string = """
    {{#system~}}
    You are an intelligent AI assistant. You will be given a question, and a passage that contains the answer to that question. Your task is to answer the question to the best of your ability using the information in the passage.
    {{~/system}}
    
    {{#user~}}
    Question: {{question}}
    
    Passage with Answer: "{{gold_passage}}"
    {{~/user}}
    
    {{#assistant~}}
    {{gen "answer" temperature=0.0 max_tokens=512}}
    {{~/assistant}}
    """
    program_string = clean_program_string(program_string)
    # Run the program
    program = guidance(program_string, llm=get_llm(model_name))  # noqa
    program_output = program(question=question, gold_passage=gold_passage)
    answer = program_output["answer"]
    # Done
    return answer


def evaluate_answer_with_debate(
    question: str,
    gold_passage: str,
    retrieved_answer: Optional[str],
    open_book_answer: str,
    closed_book_answer: str,
    model_name="gpt-4",
) -> Dict[str, Dict[str, int]]:
    program_string = """
    {{#system~}}
    Bobby and Michael are college professors co-teaching a class on Machine Learning. It's the end of semester, and their three students took the final exams. Bobby and Michael must now grade their exams. 

    There are no absolute grades for their class, meaning that the grades are relative. Bobby and Michael must therefore collectively decide how a student's answer ranks along several dimensions, namely:
    1. Comprehensiveness.
    2. Correctness.
    3. Simplicity. 
    4. Relevance. 
    
    You will be given the question, a passage containing the true gold answer ("gold passage"), and the answers of each student. Your task is to simulate a constructive argument between Bobby and Michael, as they deliberate how to rank each student along these dimensions. It's important for you to note that there cannot be ties. When they are done, they will produce a ranking as follows: 
    
    COMPREHENSIVENESS: 
    Rank 1: <name>
    Rank 2: <name> 
    Rank 3: <name>
    
    CORRECTNESS: 
    Rank 1: <name>
    Rank 2: <name> 
    Rank 3: <name>
    
    SIMPLICITY: 
    Rank 1: <name>
    Rank 2: <name> 
    Rank 3: <name>
    
    RELEVANCE: 
    Rank 1: <name>
    Rank 2: <name> 
    Rank 3: <name>
    {{~/system}}
    
    {{#user~}}
    Question: {{question}}
    
    Gold Passage: "{{gold_passage}}"
    
    Answer of John: {{open_book_answer}}
    
    Answer of James: {{closed_book_answer}}
    
    Answer of Robert: {{retrieved_answer}}
    {{~/user}}
    
    {{#assistant~}}
    {{gen "answer" temperature=0.0 max_tokens=2048}}
    {{~/assistant}}
    """
    program_string = clean_program_string(program_string)
    # Run the program
    program = guidance(program_string, llm=get_llm(model_name))  # noqa
    excuse_answer = "I'm sorry, I cannot not answer this question. I pass."
    program_output = program(
        question=question,
        gold_passage=gold_passage,
        retrieved_answer=(
            retrieved_answer if retrieved_answer is not None else excuse_answer
        ),
        open_book_answer=open_book_answer,
        closed_book_answer=closed_book_answer,
    )
    answer = program_output["answer"]

    # Parse the answer.
    def extract_ranks(text: str) -> Dict[str, List[str]]:
        # Splitting the text into potential categories with case-insensitive matching
        potential_categories = re.split(r"([A-Za-z]+):", text, flags=re.IGNORECASE)[1:]
        potential_categories = [
            potential_categories[i : i + 2]
            for i in range(0, len(potential_categories), 2)
        ]

        result_dict = {}
        for category, ranks in potential_categories:
            # Extracting names based on "Rank n: Name" pattern
            names = re.findall(r"Rank \d+: (\w+)", ranks, flags=re.IGNORECASE)
            if names:  # Only include categories that have ranks
                result_dict[category.lower()] = names

        return result_dict

    rank_dict = extract_ranks(answer)

    # Create a set of unique names
    unique_names = set()
    for ranks in rank_dict.values():
        unique_names.update(ranks)

    # Calculate max score
    max_score = max(len(ranks) for ranks in rank_dict.values()) - 1

    # Convert ranks to scores
    score_dict = {}
    for name in unique_names:
        scores = {}
        for category, ranks in rank_dict.items():
            # If the name is in the ranks, then score is max_score - rank
            # If the name is not in the ranks, then score is None
            score = max_score - ranks.index(name) if name in ranks else None
            scores[category] = score
        score_dict[name] = scores

    # Map the fake names back to the real names
    name_map = {
        "John": "open_book",
        "James": "closed_book",
        "Robert": "retrieved",
    }
    score_dict = {name_map[name]: scores for name, scores in score_dict.items()}
    return score_dict


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

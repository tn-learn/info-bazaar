import copy
import hashlib
import os
import sqlite3
import pickle

import numpy as np
import torch
from collections import defaultdict
from typing import (
    List,
    Optional,
    SupportsFloat,
    Dict,
    Any,
    Union,
    TYPE_CHECKING,
    Tuple,
)
from collections.abc import Mapping, Sequence

import types

import guidance
import re
import openai
import platformdirs
import backoff

from bazaar.py_utils import DiskCache
from llamapi.client import ask_for_guidance

if TYPE_CHECKING:
    import torch
    from bazaar.schema import Quote, Query, Answer

OAI_EXCEPTIONS = (
    openai.error.APIError,
    openai.error.RateLimitError,
    openai.error.ServiceUnavailableError,
    openai.error.TryAgain,
)

MODEL_CACHE = {}
OAI_MODELS = ["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4", "gpt-4-32k"]
OAI_EMBEDDINGS = ["text-embedding-ada-002"]

DEFAULT_LLM_NAME = "RemoteLlama-2-70b-chat-hf"
DEFAULT_RERANKER_NAME = "ms-marco-MiniLM-L-4-v2"
DEFAULT_EMBEDDING_NAME = "text-embedding-ada-002"
EMBEDDING_MANAGER = None


def default_llm_name(set_to: Optional[str] = None) -> str:
    global DEFAULT_LLM_NAME
    if set_to is not None:
        DEFAULT_LLM_NAME = set_to
    if DEFAULT_LLM_NAME is None:
        raise ValueError("Default LLM not set")
    return DEFAULT_LLM_NAME


def default_reranker_name(set_to: Optional[str] = None) -> str:
    global DEFAULT_RERANKER_NAME
    if set_to is not None:
        DEFAULT_RERANKER_NAME = set_to
    if DEFAULT_RERANKER_NAME is None:
        raise ValueError("Default reranker not set")
    return DEFAULT_RERANKER_NAME


def default_embedding_name(set_to: Optional[str] = None) -> str:
    global DEFAULT_EMBEDDING_NAME
    if set_to is not None:
        DEFAULT_EMBEDDING_NAME = set_to
    if DEFAULT_EMBEDDING_NAME is None:
        raise ValueError("Default embedding not set")
    return DEFAULT_EMBEDDING_NAME


def get_hf_auth_token(
    hf_auth_token: Optional[str], raise_if_not_found: bool = False
) -> str:
    if hf_auth_token is None:
        hf_auth_token = os.getenv("HF_AUTH_TOKEN")
    if hf_auth_token is None and raise_if_not_found:
        raise ValueError(
            "HuggingFace auth token not provided (set with export HF_AUTH_TOKEN=...)"
        )
    return hf_auth_token


def set_hf_auth_token(hf_auth_token: Optional[str] = None):
    os.environ["HF_AUTH_TOKEN"] = hf_auth_token
    return hf_auth_token


def get_hf_cache_directory(
    hf_cache_directory: Optional[str], raise_if_not_found: bool = False
) -> str:
    if hf_cache_directory is None:
        hf_cache_directory = os.getenv("HF_CACHE_DIRECTORY")
    if hf_cache_directory is None and raise_if_not_found:
        raise ValueError(
            "HuggingFace cache directory not provided (set with export HF_CACHE_DIRECTORY=...)"
        )
    if hf_cache_directory is None:
        # Use the default cache directory
        hf_cache_directory = platformdirs.user_cache_dir("huggingface")
    return hf_cache_directory


def set_hf_cache_directory(hf_cache_directory: Optional[str] = None):
    if hf_cache_directory is not None:
        os.makedirs(hf_cache_directory, exist_ok=True)
    os.environ["HF_CACHE_DIRECTORY"] = hf_cache_directory
    return hf_cache_directory


def resolve_llm_model_id(model_id: str) -> str:
    if os.environ.get("HF_LLM_MODEL_ID_OVERRIDE") is not None:
        retval = os.environ["HF_LLM_MODEL_ID_OVERRIDE"]
        assert os.path.exists(retval), f"Model path {retval} does not exist"
        return retval
    else:
        return model_id


def resolve_embedding_model_id(model_id: str) -> str:
    if os.environ.get("HF_EMBEDDING_MODEL_ID_OVERRIDE") is not None:
        retval = os.environ["HF_EMBEDDING_MODEL_ID_OVERRIDE"]
        assert os.path.exists(retval), f"Model path {retval} does not exist"
        return retval
    else:
        return model_id


def get_guidance_cache_directory(
    guidance_cache_directory: Optional[str] = None,
    raise_if_not_found: bool = False,
    auto_default: bool = True,
) -> str:
    if guidance_cache_directory is None:
        guidance_cache_directory = os.getenv("GUIDANCE_CACHE_DIRECTORY")
    if guidance_cache_directory is None and raise_if_not_found:
        raise ValueError(
            "Guidance cache directory not provided (set with export GUIDANCE_CACHE_DIRECTORY=...)"
        )
    if guidance_cache_directory is None and auto_default:
        # Use the default cache directory
        guidance_cache_directory = platformdirs.user_cache_dir("guidance")
    return guidance_cache_directory


def set_guidance_cache_directory(guidance_cache_directory: Optional[str] = None):
    if guidance_cache_directory is not None:
        os.makedirs(guidance_cache_directory, exist_ok=True)
    os.environ["GUIDANCE_CACHE_DIRECTORY"] = guidance_cache_directory
    return guidance_cache_directory


def get_sent_tokenizer():
    import nltk

    nltk.download("punkt")
    return nltk.sent_tokenize


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
        rope_scaling: str = "dynamic:2.0",
        monitor_model: bool = False,
        **super_kwargs,
    ):
        # Init the super
        self.initialize_model(
            hf_auth_token=hf_auth_token,
            hf_cache_directory=hf_cache_directory,
            size=size,
            rope_scaling=rope_scaling,
            monitor_model=monitor_model,
        )
        super().__init__(model=self.model, tokenizer=self.tokenizer, **super_kwargs)
        # Configure the base class
        self.chat_mode = True
        guidance_cache_directory = get_guidance_cache_directory(  # noqa
            guidance_cache_directory, raise_if_not_found=False, auto_default=False
        )
        if guidance_cache_directory is not None:
            # Set a custom cache directory. This is needed for MPI cluster because
            # sqlite is borked on ~/
            self.cache = DiskCache(guidance_cache_directory, self.llm_name)
        self.llm_name = self.model_id.split("/")[-1]

    def initialize_model(
        self,
        hf_auth_token: str,
        hf_cache_directory: str,
        size: str,
        rope_scaling: str,
        monitor_model: bool,
    ):
        import transformers
        import torch

        # Get the huggingface auth token if not provided
        hf_auth_token = get_hf_auth_token(hf_auth_token, raise_if_not_found=True)
        # Get the cache directory
        hf_cache_directory = get_hf_cache_directory(
            hf_cache_directory, raise_if_not_found=False
        )
        # Build the model
        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        assert size in ["7b", "13b", "70b"]
        self.model_id = f"meta-llama/Llama-2-{size}-chat-hf"

        # Configure the model
        model_id = resolve_llm_model_id(self.model_id)
        extra_config = {}

        if rope_scaling == "none":
            pass
        elif rope_scaling.startswith("dynamic"):
            factor = float(rope_scaling.split(":")[-1])
            extra_config["rope_scaling"] = {"type": "dynamic", "factor": factor}
        else:
            raise ValueError(f"Unknown rope scaling {rope_scaling}")

        # Build
        model_config = transformers.AutoConfig.from_pretrained(
            model_id,
            use_auth_token=hf_auth_token,
            cache_dir=hf_cache_directory,
            **extra_config,
        )
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            config=model_config,
            trust_remote_code=True,
            quantization_config=bnb_config,
            device_map="auto",
            use_auth_token=hf_auth_token,
            cache_dir=hf_cache_directory,
        )
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_id, use_auth_token=hf_auth_token, cache_dir=hf_cache_directory
        )
        # Patch monitor if required
        if monitor_model:
            self.model_monitor = self.patch_generate_with_input_monitor_(self.model)
        else:
            self.model_monitor = None

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


class RemoteLLM:
    use_remote_guidance: bool = True

    def get_json_identifier(self):
        raise NotImplementedError


class RemoteLlaMa2(RemoteLLM):
    def __init__(self, size: str):
        self.size = size

    def get_json_identifier(self):
        return {"model_name": f"Llama-2-{self.size}-chat-hf"}


def get_llm(model_name: Optional[str] = None, **extra_kwargs):
    if model_name is None:
        model_name = default_llm_name()
    name_to_cls_kwargs_mapping = {
        "Llama-2-70b-chat-hf": (LLaMa2, {"size": "70b"}),
        "RemoteLlama-2-70b-chat-hf": (RemoteLlaMa2, {"size": "70b"}),
    }
    if model_name in OAI_MODELS:
        llm = guidance.llms.OpenAI(model_name, **extra_kwargs)
        guidance_cache_directory = get_guidance_cache_directory(auto_default=False)
        if guidance_cache_directory is not None:
            llm.cache = DiskCache(guidance_cache_directory, llm.llm_name)
        return llm
    elif model_name in name_to_cls_kwargs_mapping:
        global MODEL_CACHE
        if model_name not in MODEL_CACHE:
            cls, kwargs = name_to_cls_kwargs_mapping[model_name]
            MODEL_CACHE[model_name] = cls(**kwargs, **extra_kwargs)
        return MODEL_CACHE[model_name]
    else:
        raise ValueError(f"Unknown model {model_name}")


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
        # Get tokens and cachedirs
        hf_cache_directory = get_hf_cache_directory(hf_cache_directory)
        hf_auth_token = get_hf_auth_token(hf_auth_token)
        # Init the tokenizer and embedding
        # Resolve model id
        model_id = resolve_embedding_model_id(model_id)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=hf_cache_directory,
            use_auth_token=hf_auth_token,
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


class LMReranker:
    def __init__(
        self,
        model_id: str,
        hf_auth_token: Optional[str] = None,
        hf_cache_directory: Optional[str] = None,
        device: str = "auto",
        max_batch_size: int = 32,
        max_num_tokens: int = 512,
    ):
        import torch
        import transformers

        # Privates
        if device == "auto":
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.model_id = model_id
        self.max_batch_size = max_batch_size
        self.max_num_tokens = max_num_tokens
        # Get tokens and cachedirs
        hf_auth_token = get_hf_auth_token(hf_auth_token)
        hf_cache_directory = get_hf_cache_directory(hf_cache_directory)
        # Init the tokenizer and model
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=hf_cache_directory,
            use_auth_token=hf_auth_token,
        )
        self.model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_id,
            cache_dir=hf_cache_directory,
            use_auth_token=hf_auth_token,
        )
        self.model.to(self.device)
        self.model.eval()

    def encode_batch(
        self, paired_queries: List[str], paired_passages: List[str]
    ) -> List[float]:
        import torch

        # Tokenize
        encoded_input = self.tokenizer(
            paired_queries,
            paired_passages,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_num_tokens,
        )
        encoded_input = to_device(encoded_input, self.device)
        with torch.no_grad():
            scores = self.model(**encoded_input).logits
        # Move back to cpu
        scores = to_device(scores, "cpu")
        # Convert to list
        if scores.dim() == 2:
            # Remove singleton dimension
            assert scores.shape[-1] == 1, f"Unexpected shape: {scores.shape}"
            scores = scores[:, 0]
        # Convert to list and return
        return scores.tolist()

    def encode(
        self, query: Union[str, List[str]], passages: Union[str, List[str]]
    ) -> List[List[float]]:
        # Convert to lists
        if isinstance(query, str):
            query = [query]
        if isinstance(passages, str):
            passages = [passages]
        # Build all pairs
        pairs = []
        for q in query:
            for p in passages:
                pairs.append((q, p))
        paired_queries, paired_passages = zip(*pairs)
        # Split to batches
        paired_query_batches = [
            paired_queries[i : i + self.max_batch_size]
            for i in range(0, len(paired_queries), self.max_batch_size)
        ]
        paired_passage_batches = [
            paired_passages[i : i + self.max_batch_size]
            for i in range(0, len(paired_passages), self.max_batch_size)
        ]
        # Run it in a loop
        scores = []
        for q, p in zip(paired_query_batches, paired_passage_batches):
            scores.extend(self.encode_batch(q, p))
        assert len(scores) == len(pairs)
        # Reshape to a matrix
        scores = [
            scores[i : i + len(passages)] for i in range(0, len(scores), len(passages))
        ]
        assert len(scores) == len(query)
        return scores


class CrossEncoderMiniLMReranker(LMReranker):
    def __init__(self, num_layers: int, **super_kwargs):
        assert num_layers in [2, 4, 6, 12]
        model_id = f"cross-encoder/ms-marco-MiniLM-L-{num_layers}-v2"
        super().__init__(model_id, **super_kwargs)


def get_reranker(model_name: Optional[str] = None, **extra_kwargs):
    if model_name is None:
        model_name = default_reranker_name()
    name_to_cls_and_kwargs_mapping = {
        "ms-marco-MiniLM-L-2-v2": (CrossEncoderMiniLMReranker, {"num_layers": 2}),
        "ms-marco-MiniLM-L-4-v2": (CrossEncoderMiniLMReranker, {"num_layers": 4}),
        "ms-marco-MiniLM-L-6-v2": (CrossEncoderMiniLMReranker, {"num_layers": 6}),
        "ms-marco-MiniLM-L-12-v2": (CrossEncoderMiniLMReranker, {"num_layers": 12}),
    }
    if model_name in name_to_cls_and_kwargs_mapping:
        global MODEL_CACHE
        if model_name not in MODEL_CACHE:
            cls, kwargs = name_to_cls_and_kwargs_mapping[model_name]
            MODEL_CACHE[model_name] = cls(**kwargs, **extra_kwargs)
        return MODEL_CACHE[model_name]
    else:
        raise ValueError(f"Unknown model {model_name}")


class EmbeddingManager:
    def __init__(self, db_path: str, raise_if_cache_miss: bool = False):
        self.raise_if_cache_miss = raise_if_cache_miss
        self.conn = sqlite3.connect(db_path)
        self._create_table()

    def _create_table(self) -> None:
        with self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    key TEXT PRIMARY KEY,
                    embedding BLOB
                )
            """
            )

    def _create_key(self, text: str, model_name: Optional[str] = None) -> str:
        if model_name is None:
            model_name = default_embedding_name()
        return hashlib.sha256((text + model_name).encode()).hexdigest()

    def get_embedding(
        self,
        content: str,
        model_name: Optional[str] = None,
        generate_if_missing: bool = True,
    ) -> Optional[List[float]]:
        if model_name is None:
            model_name = default_embedding_name()

        key = self._create_key(content, model_name)
        cursor = self.conn.execute(
            "SELECT embedding FROM embeddings WHERE key = ?", (key,)
        )
        result = cursor.fetchone()
        if result:
            return self._bytes_to_embedding(result[0])
        elif generate_if_missing:
            return self.new_embedding(content, model_name, key)
        elif self.raise_if_cache_miss:
            raise ValueError(
                f"Embedding not found with model {model_name} and content:\n{content}"
            )
        else:
            return None

    def new_embedding(
        self, content: str, model_name: str, key: Optional[str] = None
    ) -> List[float]:
        if key is None:
            key = self._create_key(content, model_name)
        computed_embedding = generate_embedding(content, model=model_name)
        self._store_embedding(key, computed_embedding)
        return computed_embedding

    def _store_embedding(self, key: str, embedding: List[float]) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO embeddings (key, embedding) VALUES (?, ?)",
                (key, self._embedding_to_bytes(embedding)),
            )

    def _bytes_to_embedding(self, bytes_: bytes) -> List[float]:
        return [float(e) for e in pickle.loads(bytes_)]

    def _embedding_to_bytes(self, embedding: List[float]) -> bytes:
        return pickle.dumps([float(e) for e in embedding])

    def close(self):
        self.conn.close()
        return self

    def build_index(
        self, texts: List[str], model_name: Optional[str] = None, use_tqdm: bool = False
    ) -> "EmbeddingManager":
        if use_tqdm:
            from tqdm import tqdm

            texts = tqdm(texts)
        for text in texts:
            self.get_embedding(text, model_name=model_name)
        return self


def global_embedding_manager(
    init_from_path: Optional[str] = None, **embedding_manager_kwargs
) -> "EmbeddingManager":
    global EMBEDDING_MANAGER
    if init_from_path is not None:
        EMBEDDING_MANAGER = EmbeddingManager(init_from_path, **embedding_manager_kwargs)
    if EMBEDDING_MANAGER is None:
        raise ValueError("Embedding manager not initialized.")
    return EMBEDDING_MANAGER


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


@backoff.on_exception(backoff.expo, OAI_EXCEPTIONS, max_tries=5)
def break_down_question(question: str, model: Optional[str] = None) -> List[str]:
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
    """  # noqa
    program_string = clean_program_string(program_string)

    program_outputs = ask_for_guidance(
        program_string=program_string,
        llm=get_llm(model),
        silent=True,
        inputs=dict(question=question),
        output_keys=["subqs"],
    )
    return _extract_questions(program_outputs["subqs"])


@backoff.on_exception(backoff.expo, OAI_EXCEPTIONS, max_tries=5)
def generate_hyde_passage(question: str, model: Optional[str] = None) -> str:
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
    program_outputs = ask_for_guidance(
        program_string=program_string,
        llm=get_llm(model),
        silent=True,
        inputs=dict(
            question=question,
            parse_answer=_parse_answer,
        ),
        output_keys=["hyde_answer"],
    )
    return program_outputs["hyde_answer"]


def generate_keywords(text: str, model_name: Optional[str] = None) -> List[str]:
    program_string = """
    {{#system~}}
    You will be given some text, which may be a passage or a question. Your task is to extract keywords that can be useful for search. 

    The output must be comma separated keywords, as in: "first keyword, second keyword, ..."
    {{~/system}}
    
    {{#user~}}
    Text: {{text_to_keywordify}}
    {{~/user}}
    
    {{#assistant~}}
    {{gen 'keywords' stop="\\n" temperature=0.0}}
    {{~/assistant}}
    """
    program_string = clean_program_string(program_string)
    program_outputs = ask_for_guidance(
        program_string=program_string,
        llm=get_llm(model_name),
        silent=True,
        inputs=dict(text_to_keywordify=text),
        output_keys=["keywords"],
    )
    # Split keywords by comma
    keywords = program_outputs["keywords"].split(",")
    # Clean up
    keywords = [keyword.strip() for keyword in keywords]
    # Remove empty keywords
    keywords = [keyword for keyword in keywords if keyword != ""]
    return keywords


@backoff.on_exception(backoff.expo, OAI_EXCEPTIONS, max_tries=5)
def generate_embedding(
    text: str,
    model: Optional[str] = None,
    as_query: bool = False,
    embedding_manager: Optional["EmbeddingManager"] = None,
    cache_if_embedding_manager_available: bool = False,
    **embedder_kwargs,
) -> List[float]:
    if model is None:
        model = default_embedding_name()
    if embedding_manager is None:
        embedding_manager = EMBEDDING_MANAGER
    # Check if we have a cached embedding
    if embedding_manager is not None:
        embedding = embedding_manager.get_embedding(
            text, model_name=model, generate_if_missing=False
        )
        if embedding is not None:
            return embedding
    # We'll need to generate an embedding
    if model in OAI_EMBEDDINGS:
        embedding = openai.Embedding.create(
            input=[text], model=model, **embedder_kwargs
        )["data"][0]["embedding"]
    else:
        # Get huggingface embedder
        embedder = get_embedder(model, **embedder_kwargs)
        # Get the embedding
        embedding = embedder.encode(text, as_query=as_query)
    # Cache the embedding to database if required
    if embedding_manager is not None and cache_if_embedding_manager_available:
        embedding_manager.new_embedding(text, model)
    return embedding


@backoff.on_exception(backoff.expo, OAI_EXCEPTIONS, max_tries=5)
def split_to_paragraphs(
    block: "Block", model_name: str, target_num_paragraphs: int = -1
) -> List["Block"]:

    if model_name in OAI_MODELS:
        raise NotImplementedError("Deep guidance not implemented for OpenAI models.")
    if target_num_paragraphs == -1:
        target_num_paragraphs = (block.num_tokens // 450) + 1
    if target_num_paragraphs == 1:
        return [block]
    # TODO: fix the length issue then delete this.
    if block.num_tokens > 450:
        block.content = block.content[:450]
        return [block]
    # Split text by sentences
    text = block.content
    sentences = get_sent_tokenizer()(text)
    # This one's for the llamas
    program_string = """
    {{#system~}}
    You are a text formatting bot. You will be provided with an ordered list of sentences. Your task is to group these sentences in to paragraphs. Each paragraph should be as self-contained as possible, meaning they should be understandable independently of the other paragraphs. 

    You will first reflect about the provided sentences by writing a few sentences about how you plan to proceed. Once you are done, you will print a list: 

    Sentence 1: Paragraph <paragraph index>
    Sentence 2: Paragraph <paragraph index>
    ... and so on. 

    For example, let's say you are given 5 sentences and you have to split them in to 2 paragraphs. You might want to put the first three sentences in the first paragraph and the last two in the second paragraph. In this case, you would output: 

    <few sentences about how you want to put the first three sentences in the first paragraph and the last two in the second paragraph>

    Sentence 1: Paragraph 1
    Sentence 2: Paragraph 1
    Sentence 3: Paragraph 1
    Sentence 4: Paragraph 2
    Sentence 5: Paragraph 2
    {{~/system}}

    {{#user~}}
    Here are the sentences that you are given. 
    {{#each sentences}}
    Sentence {{add @index 1}}: {{this}}{{/each}}

    You must split these sentences to {{num_para}} paragraphs. You're up. 
    {{~/user}}

    {{#assistant~}}
    Understood, let us reflect about these sentences in a paragraph. 

    {{gen 'thinks' temperature=0.1 stop="\\n\\n" max_tokens=512}}

    Here is the list of sentences with their corresponding paragraph numbers.
    {{#each sentences}}
    Sentence {{add @index 1}}: Paragraph {{gen 'parasplits' list_append=True stop='\\n'}}{{/each}}
    {{~/assistant}}
    """
    program_string = clean_program_string(program_string)
    program_output = ask_for_guidance(
        program_string=program_string,
        llm=get_llm(model_name=model_name),
        silent=True,
        inputs=dict(
            sentences=sentences,
            num_para=target_num_paragraphs,
        ),
        output_keys=["parasplits"],
    )
    paragraph_indices = [int(x) - 1 for x in program_output["parasplits"]]
    num_paragraphs = len(set(paragraph_indices))
    # Split the sentences into paragraphs as given by the paragraph indices
    paragraphs = [
        [
            sentences[para_idx]
            for para_idx in paragraph_indices
            if para_idx == current_para_idx
        ]
        for current_para_idx in range(num_paragraphs)
    ]
    # Join the sentences in each paragraph
    paragraphs = [" ".join(paragraph) for paragraph in paragraphs]
    ret_blocks = []
    token_start = block.token_start
    for paragraph in paragraphs:
        block_cpy = copy.deepcopy(block)
        block_cpy.content = paragraph
        block_cpy.token_start = token_start
        block_cpy.token_end = token_start + block_cpy.num_tokens
        ret_blocks.append(block_cpy)
        token_start = block_cpy.token_end
    return ret_blocks


@backoff.on_exception(backoff.expo, OAI_EXCEPTIONS, max_tries=5)
def select_quotes_with_heuristic(
    quotes: List["Quote"],
    budget: Optional[SupportsFloat] = None,
    fraction_of_max_budget: Optional[float] = None,
    model_name: Optional[str] = None,
) -> List["Quote"]:
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
    program_output = ask_for_guidance(
        program_string=program_string,
        llm=get_llm(model_name=model_name),
        silent=True,
        inputs=dict(
            question=question,
            options=options,
            balance=budget,
            average_quote_price=average_quote_price,
        ),
        output_keys=["answer"],
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


@backoff.on_exception(backoff.expo, OAI_EXCEPTIONS, max_tries=5)
def extract_reasonable_questions_from_passage(
    passage: str, model_name: Optional[str] = None
) -> List[str]:
    program_string = """
    {{#system~}}
    Bobby is an exam creator and Michael is an exam auditor. Bobby's job is to read a passage and propose some questions that could be answered by someone who has not seen that passage. Michael's job is determine whether a question is good or bad. Good questions are factual and have an objective answer. Bad questions are ambiguous, make specific references, or reference the passage in any way.

    Your task is to simulate a constructive argument between Bobby and Michael, where Bobby proposes some questions and Michael filters those questions. At the end of the argument, they must arrive at a list of good questions, indicated:
    QUESTION 1. <question>
    QUESTION 2. <question>
    and so on.
    {{~/system}}

    {{#user~}}
    The passage is: 
    {{passage}}

    Remember that a good question does not reference the passage in any way.
    {{~/user}}

    {{#assistant~}}
    {{gen "deliberation" temperature=0.1 max_tokens=2048}}
    {{~/assistant}}
    """
    program_string = clean_program_string(program_string)
    program_output = ask_for_guidance(
        program_string=program_string,
        llm=get_llm(model_name=model_name),
        silent=True,
        inputs=dict(passage=passage),
        output_keys=["deliberation"],
    )
    # Extract questions
    def extract_questions(text):
        lines = text.split("\n")
        questions = []

        for line in lines:
            match = re.match(r"QUESTION \d+\.\s*(.*)", line)
            if match:
                questions.append(match.group(1).strip())

        return questions

    extracted_questions = extract_questions(program_output["deliberation"])
    return extracted_questions


@backoff.on_exception(backoff.expo, OAI_EXCEPTIONS, max_tries=5)
def extract_questions(content: str, model_name: Optional[str] = None) -> List[str]:
    program_string = """
    {{#system~}}
    Bobby is an exam creator and Michael is an exam auditor. Bobby's job is to read a passage and propose some questions that could be answered by someone who has not seen that passage. Michael's job is determine whether a question is good or bad. Good questions are factual and have an objective answer. Bad questions are ambiguous, make specific references, or reference the passage in any way.
    
    Your task is to simulate a constructive argument between Bobby and Michael, where Bobby proposes some questions and Michael filters those questions. At the end of the argument, they must arrive at a list of good questions, indicated:
    QUESTION 1. <question>
    QUESTION 2. <question>
    and so on.
    {{~/system}}
    
    {{#user~}}
    The passage is: 
    {{passage}}
    
    Remember that a good question does not reference the passage in any way.
    {{~/user}}
    
    {{#assistant~}}
    Bobby: Alright, let's get started with the deliberation. Here's my first question:
    {{gen "deliberation" temperature=0.1 max_tokens=1024}}
    
    Bobby: Great! So, we have the following good questions:
    {{gen "questions" temperature=0.1 max_tokens=1024}}
    {{~/assistant}}
    """
    program_string = clean_program_string(program_string)
    program_output = ask_for_guidance(
        program_string=program_string,
        llm=get_llm(model_name=model_name),
        silent=True,
        inputs=dict(passage=content),
        output_keys=["questions"],
    )
    pattern = re.compile(r"QUESTION \d+\. (.+?)\n")
    try:
        questions = pattern.findall(program_output["questions"])
    except Exception:
        questions = []
    return questions


@backoff.on_exception(backoff.expo, OAI_EXCEPTIONS, max_tries=5)
def select_quotes_with_debate(
    quotes: List["Quote"],
    budget: Optional[SupportsFloat] = None,
    fraction_of_max_budget: Optional[float] = None,
    model_name: Optional[str] = None,
    use_block_content_metadata: bool = False,
) -> List["Quote"]:
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
                [
                    (
                        block.content_with_metadata
                        if use_block_content_metadata
                        else block.content
                    )
                    for block in quote.answer_blocks
                ]
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
    program_output = ask_for_guidance(
        program_string=program_string,
        llm=get_llm(model_name=model_name),
        silent=True,
        inputs=dict(
            question=question,
            options=options,
            balance=100,
        ),
        output_keys=["answer"],
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


@backoff.on_exception(backoff.expo, OAI_EXCEPTIONS, max_tries=5)
def clean_content(content: str, model_name: Optional[str] = None) -> str:
    program_string = """
    {{#system~}}
    You are a text passage cleaner bot. You will be provided a text passage and you will reply with an exact copy of the input text passage, but with the following (and only the following) modifications:

    1. Improve use of white space, tabs, and new-lines. This should shorten the passage.
    2. Remove citations (e.g., Huges et al. [hugesGenerativeAdversarialLearning]).
    3. Reformat any tabular data into markdown.
    {{~/system}}
    
    {{#user~}}
    The text passage is: {{passage}}
    {{~/user}}
    
    {{#assistant~}}
    Sure, here is what I'm going to change:
    {{gen "rationale" temperature=0.1}}
    
    Here is the passage with cleaned text:
    {{gen "cleaned_passage" temperature=0.1  stop="\\n\\n"}}
    {{~/assistant}}    
    """
    program_string = clean_program_string(program_string)
    program_output = ask_for_guidance(
        program_string=program_string,
        llm=get_llm(model_name=model_name),
        silent=True,
        inputs=dict(passage=content),
        output_keys=["cleaned_passage"],
    )
    cleaned_passage = program_output["cleaned_passage"]
    return cleaned_passage.strip()


def rerank_quotes(
    quotes: List["Quote"], model_name: Optional[str] = None
) -> List[float]:
    question = quotes[0].query.text
    passages = [
        " [...] ".join([block.content for block in quote.answer_blocks])
        for quote in quotes
    ]
    reranker = get_reranker(model_name)
    # Get the scores for the first query, which is the question
    scores = reranker.encode(question, passages)[0]
    return scores


@backoff.on_exception(backoff.expo, OAI_EXCEPTIONS, max_tries=5)
def synthesize_answer(
    query: "Query", quotes: List["Quote"], model_name: Optional[str] = None
) -> str:
    question = query.text
    # Make sure all quotes have the same query
    assert all(
        [query.compare_content(quote.query) for quote in quotes]
    ), "All quotes must have the same query."

    if len(quotes) == 0:
        raise ValueError("No quotes provided to synthesize answers from.")

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
    You are an AnswerSynthesisBot. Your task is to synthesize an answer to a question given some passages that should contain the answer. You will combine and synthesize the information provided to you. Your answer should be understandable and to the point. 

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
    program_output = ask_for_guidance(
        program_string=program_string,
        llm=get_llm(model_name=model_name),
        silent=True,
        inputs=dict(
            question=question,
            quotes=passages,
        ),
        output_keys=["answer"],
    )

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

    answer = separate_text_to_dict_corrected(program_output["answer"])["answer"]
    return answer


@backoff.on_exception(backoff.expo, OAI_EXCEPTIONS, max_tries=5)
def get_closed_book_answer(question: str, model_name: Optional[str] = None) -> str:
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
    program_output = ask_for_guidance(
        program_string=program_string,
        llm=get_llm(model_name=model_name),
        silent=True,
        inputs=dict(question=question),
        output_keys=["answer"],
    )
    answer = program_output["answer"]
    # Done
    return answer


def synthesize_compound_answer(
    query_answer_pairs: List[Tuple["Query", "Answer"]], model_name: Optional[str] = None
) -> str:
    # TODO
    pass


@backoff.on_exception(backoff.expo, OAI_EXCEPTIONS, max_tries=5)
def get_open_book_answer(
    question: str, gold_passage: str, model_name: Optional[str] = None
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
    program_output = ask_for_guidance(
        program_string=program_string,
        llm=get_llm(model_name=model_name),
        silent=True,
        inputs=dict(question=question, gold_passage=gold_passage),
        output_keys=["answer"],
    )
    answer = program_output["answer"]

    return answer


def select_follow_up_question(
    question: str, current_answer: str, model_name: Optional[str] = None
) -> Optional[str]:
    program_string = """
    {{#system~}}
    Bobby and Michael are employed at a company that specializes in acquiring and verifying information.

    Their supervisors have given them a question and an answer that their peers have produced. Their task is to decide if the provided answer adequately answers the question. If it doesn't, they must come up with a follow up question that would enrich the provided answer. 

    Note that the follow up question should only be asked if there is a need for concrete information that is missing from the provided answer or if the provided answer is missing crucial details.
    {{~/system}}

    {{#user~}}
    The question is: {{question}}

    The currently available answer is: {{current_answer}}

    Bobby and Michael will now argue about whether they should ask a follow-up question taking in to account the provided question and the currently available answer. 

    If they decide to ask a follow up question, it should be printed as:
    FOLLOW-UP QUESTION: <follow up question goes here>
    {{~/user}}

    {{#assistant~}}
    {{gen "answer" temperature=0.0 max_tokens=512}}
    {{~/assistant}}
    """
    program_string = clean_program_string(program_string)
    # Run the program
    program_output = ask_for_guidance(
        program_string=program_string,
        llm=get_llm(model_name=model_name),
        silent=True,
        inputs=dict(question=question, current_answer=current_answer),
        output_keys=["answer"],
    )
    answer = program_output["answer"]

    # Parse the answer
    def extract_follow_up(text: str) -> Optional[str]:
        # Define the pattern
        pattern = r"FOLLOW-UP QUESTION: (.*?)(\n|$)"
        # Search for the pattern
        match = re.search(pattern, text)
        # If found, return the matched question, else return None
        return match.group(1).strip() if match else None

    follow_up = extract_follow_up(answer)
    return follow_up


@backoff.on_exception(backoff.expo, OAI_EXCEPTIONS, max_tries=5)
def evaluate_answer_with_debate(
    question: str,
    gold_passage: str,
    retrieved_answer: Optional[str],
    open_book_answer: str,
    closed_book_answer: str,
    model_name: str = "gpt-4",
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
    excuse_answer = "I'm sorry, I cannot not answer this question. I pass."
    program_output = ask_for_guidance(
        program_string=program_string,
        llm=get_llm(model_name=model_name),
        silent=True,
        inputs=dict(
            question=question,
            gold_passage=gold_passage,
            retrieved_answer=(
                retrieved_answer if retrieved_answer is not None else excuse_answer
            ),
            open_book_answer=open_book_answer,
            closed_book_answer=closed_book_answer,
        ),
        output_keys=["answer"],
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
    # TODO: assert only names in name_map present
    score_dict.pop("None", None)
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

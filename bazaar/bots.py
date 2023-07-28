import os
import json
import uuid

import backoff as backoff
import openai
import pickle
import tiktoken
import textwrap
import numpy as np
from tqdm import tqdm
import streamlit as st
import json
from typing import List
from pathlib import Path
from datetime import datetime

JSON_INDENT = 2
try:
    openai.api_key = os.environ.get("OPENAI_API_KEY")
except Exception as e:
    print("Failed to set OpenAI API key.")

director_bot_system_prompt = """You are DirectorBot, an AI agent directing a team of other AI analyst agents in order to find information requested by a client. You drive the investigation by asking specific questions, and the AI analyst agents will try to answer them. Note that the analysts may make mistakes or fail if your question is too broad or vague, so you must make sure to ask it specific questions. It is also extremely important for you to communicate only via JSON blobs.  Here's how this works.

1. You will be asked a question by the client. The client question will be formatted as the following JSON blob: `{"message_from": "client", "message_to": "DirectorBot", "message_content": "<content of the message>"}`.

2. You will devise a strategy to answer that question. It tends to be useful to decompose the question in to sub-questions, but you be the judge. Either way, you are encouraged to write down your thoughts in detail by sending messages to yourself, as in `[{"message_from": "DirectorBot", "message_to": "DirectorBot", "message_content": "<thought 1>"}, {"message_from": "DirectorBot", "message_to": "DirectorBot", "message_content": "<thought 2>"}, ...]`. After you have devised the strategy by sending JSON blob to yourself, you might need extra information. In this case, you will task one or more AI analyst agents to go find the answer for you. The analyst will look for information in a database, but you will have to tell it what to look for via a search phrase. These messages look like: `[{"message_from": "DirectorBot", "message_to": "AnalystBot_0", "message_content": "<your question here>", "search_phrase": "<search phrase goes here>"}, {"message_from": "DirectorBot", "message_to": "AnalystBot_1", "message_content": "<your question here>", "search_phrase": "<search phrase goes here>"}, ...]`. The analyst agents will respond to you once they have completed their jobs. Their response will look like `{"message_from": "AnalystBot_0", "message_to": "DirectorBot", "message_content": "<content of the message>", "success": true, "used_documents": ["<id of used document 1>", ...]}`. Note that they might fail to find the information, in which case the `"success"` field will be set to `false`. If this happens, you will need to adapt.

3. You will combine and synthesize the information provided to you by the analyst agents in a way that helps answer the client's question. Should you need it, you have a python shell at your disposal. It can be useful for complex calculations. You may use it as `[{"message_from": "DirectorBot", "message_to": "python", "message_content": "<python snippet goes here>"}]`

4. You will respond to the client. These messages look like `{"message_from": "DirectorBot", "message_to": "client", "message_content": [{"sentence": <sentence_1>, "used_documents": ["<id of used document 1>", ...]}], ...}`. Here, the value of message_content is a list of dictionaries, where each dictionary contains a sentence and an optional list of document ids used to write that sentence.

Let's begin."""

analyst_bot_system_prompt = """You are an AnalystBot, an AI analyst agent tasked with answering a question provided to you, given some documents from a database. Your task is to inspect the document and to try to answer the question.  If the answer cannot be found or inferred, your task is to communicate that (see below). It is extremely important for you to communicate only via JSON blobs.  Here's how this works.

1. You will be asked a question by the client. This question will look like `{"message_from": "client", "message_to": "AnalystBot", "message_content": "<question>"}`, and you will try to answer the question based on the documents provided to you. These documents will be provided to you as `[{"message_from": "database", "message_content": "<content of the document>", "document_id": <id of the document>, "acquisition_date": "<date document was accquired>", "document_title": "<title of document if available>"}, ...]`. It is possible that some documents are outdated, and you will need to adapt accordingly.

2. Begin your response with a list of salient facts present in the text. This will help you think step-by-step. You will respond as `[{"message_from": "AnalystBot", "message_to": "AnalystBot", message_content: "<relevant fact 1>"}, {"message_from": "AnalystBot", "message_to": "AnalystBot", message_content: "<relevant fact 2>"}, ...]`.  

3. If you have found the answer, you will respond with `{"message_from": "AnalystBot", "message_to": "client", "message_content": "<your answer>", "used_documents": ["<document id 1>", "<document id 2>", ...], "success": true}`. Note that`"used_documents"` contains the ids of the documents you used to derive your answer from. This is important to track the provenance of the information that you extract.

4. It is possible that the answer does not exist in the documents that you were provided. In this case, please respond with `{"message_from": "AnalystBot", "message_to": "client", "message_content": "<your message to the client>", "used_documents": [], "success": false}`. This will communicate with the client that the provided documents are not enough to answer the posed question.

Let's begin."""

search_bot_prompt = """ You are a SearchBot, an AI agent tasked with finding documents in a database. Your task is to find documents that are relevant to a search phrase provided to you. You will read through documents from newsapi.org to surface relevant news. It is extremely important for you to communicate only via JSON blobs.  Here's how this works.

1. You will be asked to find documents relevant to a search phrase. This will look like `{"message_from": "DirectorBot", "message_to": "SearchBot", "message_content": "<search phrase>"}`.

2. You will be presented with a list of news article titles that look like `[{"title": <title_1>, "content": <content_1>, "author": <author_1>}, ...]`. You must respond by repeating the titles that were interesting like `[{"message_from": "SearchBot", "message_to": "SearchBot", "message_content": [<title_2>, ...]}]. If none of the article titles are relevant, you may respond with an empty list as message_content. 

"""


def create_api_message(api_role, role, *messages):
    assert api_role in ["user", "system", "assistant"], (
        f"api_role must be one of 'user', 'system', or "
        f"'assistant', but got {api_role}"
    )
    
    return {"role": api_role, "content": "".join(messages)}


def encode(text, model_name):
    return tiktoken.encoding_for_model(model_name).encode(text)


def decode(tokens, model_name):
    return tiktoken.encoding_for_model(model_name).decode(tokens)


def calculate_embedding(text):
    return openai.Embedding.create(
        input=[text.replace("\n", " ")], model="text-embedding-ada-002"
    )["data"][0]["embedding"]


def chunkify(lst, chunk_size):
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def convert_to_dict(s):
    dictionary = {"keywords": []}
    s = s.strip()

    if s.startswith("KEYWORDS:"):
        elements = s.replace("KEYWORDS:", "").split(",")
        for element in elements:
            dictionary["keywords"].append(element.strip())
    elif '\n' in s:
        elements = s.split("\n")
        for element in elements:
            if element.startswith("-"):
                dictionary["keywords"].append(element[1:].strip())
            elif element[0].isdigit() and element[1] == '.':
                dictionary["keywords"].append(element[2:].strip())
    elif ',' in s:
        elements = s.split(",")
        for element in elements:
            dictionary["keywords"].append(element.strip())
    else:
        dictionary["keywords"].append(s.strip())
    dictionary["keywords"] = ", ".join(dictionary["keywords"])
    return dictionary



HYDE_PROMPT = """
Below is a question. Your task is to create a paragraph from a fictional document (e.g. a 10K filing) that exactly answers that question. This excerpt might look out of context, but that’s ok — the important bit is that it unambiguously answers the question. Respond only with the text of this fictional excerpt, and nothing else.

Question: {question}
"""


@backoff.on_exception(backoff.expo, (openai.error.RateLimitError, openai.error.ServiceUnavailableError))
def hyde(question):
    bot = Bot("hydebot", "gpt-3.5-turbo", "You are a helpful assistant.")
    bot.add_message(HYDE_PROMPT.format(question=question))
    response = bot.complete()
    # print(f"fake doc: {response}")
    return response


class Bot:
    def __init__(self, name, model_name, system_prompt):
        self.name = f"{name}_{model_name}_{str(uuid.uuid4().hex[:5])}"
        self.model_name = model_name
        self.system_prompt = create_api_message("system", None, system_prompt)
        self._history = [self.system_prompt]

    def add_message(self, message, role="user"):
        self._history.append(create_api_message(role, None, message))

    def normal_complete(self, max_tokens=750, temperature=0.0):
        completion = openai.Completion.create(
            model=self.model_name,
            prompt=self._history[0]["content"],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        response = completion.choices[0].text
        self._history.append(create_api_message("assistant", None, response))
        return response

    def complete(self, max_tokens=750, temperature=0.0):
        completion = openai.ChatCompletion.create(
            model=self.model_name,
            messages=self._history,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        response = completion.choices[0].message.content
        self._history.append(create_api_message("assistant", None, response))
        return response

    def _jsonify_history(self):
        history = []
        for event in self._history:
            event = dict(event)
            if event["role"] == "system":
                history.append(event)
            elif event["role"] in ["assistant", "user"]:
                content = event["content"]
                try:
                    content = json.loads(content)
                except Exception:
                    pass
                event["content"] = content
                history.append(event)
        return history

    def dump_history(self, root_dir, jsonify_history=True):
        path = os.path.join(
            root_dir,
            f"history_{self.name}_{datetime.now().strftime('%Y%m%d%H%M%S')}.json",
        )
        if jsonify_history:
            history = self._jsonify_history()
        else:
            history = self._history
        with open(path, "w") as f:
            json.dump(history, f, indent=2)


def chunk_texts(text, text_tokens_chunk_size, model_name):
    tokens = encode(text, model_name)
    chunked_tokens = chunkify(tokens, text_tokens_chunk_size)
    chunked_texts = [
        decode(tokens_chunk, model_name) for tokens_chunk in chunked_tokens
    ]
    return chunked_texts
from typing import Any
import pickle

import transformers


class RemoteGuidanceClient:
    """This class should look like a HF model to guidance."""

    def __init__(self, *args, **kwargs):
        args, kwargs = self.preprocess_model_args_and_kwargs(args, kwargs)
        self._model_args = args
        self._model_kwargs = kwargs
        # Init the config (which is needed downstream)
        assert "config" in kwargs, "Config must be pre-specified for RemoteGuidance."
        self.config = kwargs.get("config")
        self.device = "cpu"

    def preprocess_model_args_and_kwargs(self, args: tuple, kwargs: dict):
        # Make shallow copies
        args = tuple(args)
        kwargs = dict(kwargs)
        # Replace cache dir with an instruction to fill it in on the server
        if "cache_dir" in kwargs:
            kwargs["cache_dir"] = "env:LLAMAPI_HF_CACHE_DIRECTORY"
        return args, kwargs

    def serialize_model_args_and_kwargs(self) -> bytes:
        return pickle.dumps((self._model_args, self._model_kwargs))

    def generate(self, *args, **kwargs):
        # This is where we deflate the payload
        pass

    def prepare_inputs_for_generation(self):
        pass


def test_remote_guidance():
    model_id = "meta-llama/Llama-2-70b-chat-hf"
    model_config = transformers.AutoConfig.from_pretrained(model_id)
    model = RemoteGuidanceClient(model_config)
    pass

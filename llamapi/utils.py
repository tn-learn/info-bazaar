import re
from contextlib import contextmanager
from typing import Any, Dict, List

from guidance.llms._transformers import RegexLogitsProcessor


def getstate_llm():
    pass


def getstate_regex_logit_preprocessor(self: RegexLogitsProcessor):
    state = dict(self.__dict__)
    state["pattern_no_stop"] = self.pattern_no_stop.pattern
    state["pattern"] = self.pattern.pattern
    return state


def setstate_regex_logit_preprocessor(self: RegexLogitsProcessor, state: Dict[str, Any]):
    # Shallow copy state
    state = dict(state)
    # Replace patterns with compiled regexes
    state["pattern_no_stop"] = re.compile(state["pattern_no_stop"])
    state["pattern"] = re.compile(state["pattern"])
    # Set state
    self.__dict__.update(state)


def make_pickleable_(cls: type):
    getter = None
    setter = None
    for key, fn in globals().items():
        if not key.startswith("getstate_") or key.startswith("setstate_"):
            continue
        if getter is not None and setter is not None:
            continue
        # If we're here, we're looking at the right function
        self_type = fn.__annotations__.get("self")
        if self_type is None:
            raise ValueError(f"Function {key} should have type annotations for self.")
        if isinstance(self_type, type):
            self_type = self_type.__name__
        # Compare the names
        if cls.__name__ == self_type:
            # We have a hit.
            if key.startswith("getstate_"):
                assert getter is None, f"Getter is already set for {key}!"
                getter = fn
            if key.startswith("setstate_"):
                assert setter is None, f"Setter is already set for {key}!"
                setter = fn
    if getter is not None:
        cls._old__getstate__ = cls.__getstate__
        cls.__getstate__ = getter
    if setter is not None:
        cls._old__setstate__ = cls.__setstate__
        cls.__setstate__ = setter
    return cls


def undo_make_pickleable_(cls: type):
    if hasattr(cls, "_old__getstate__"):
        cls.__getstate__ = cls._old__getstate__
        del cls._old__getstate__
    if hasattr(cls, "_old__setstate__"):
        cls.__setstate__ = cls._old__setstate__
        del cls._old__setstate__
    return cls


@contextmanager
def make_temporarily_pickleable_(*classes: type):
    for cls in classes:
        make_pickleable_(cls)
    yield
    for cls in classes:
        undo_make_pickleable_(cls)

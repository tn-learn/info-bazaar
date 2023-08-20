import json
import os

import diskcache
import jsonpickle
import hashlib
import pickle
import git

from pathlib import Path
from typing import Any, Union, SupportsFloat

from guidance.llms.caches import Cache

PathType = Union[str, Path]


def root_dir_slash(path: str) -> str:
    # Get the root dir of the repo where this file lives
    repo_root = git.Repo(__file__, search_parent_directories=True).working_tree_dir
    path = Path(repo_root) / path
    if not path.exists():
        path.mkdir(exist_ok=True, parents=True)
    return str(path)


def dump_json(obj: Any, path: PathType):
    with open(str(path), "w") as f:
        frozen = jsonpickle.encode(obj)
        f.write(frozen)


def load_json(path: PathType) -> Any:
    with open(str(path), "r") as f:
        thawed = jsonpickle.decode(f.read())
    return thawed


def dump_dict(d: dict, path: PathType) -> dict:
    with open(str(path), "w") as f:
        json.dump(d, f, indent=2)
    return d


def load_dict(path: PathType) -> dict:
    with open(str(path), "r") as f:
        d = json.load(f)
    return d


def ensure_number(
    x: Union[SupportsFloat], allow_none: bool = False
) -> Union[int, float, None]:
    import numpy as np
    import torch

    if isinstance(x, int):
        return x
    elif isinstance(x, float):
        return x
    elif isinstance(x, np.ndarray):
        assert x.size == 1
        return x.item()
    elif isinstance(x, torch.Tensor):
        assert x.numel() == 1
        return x.item()
    elif x is None and allow_none:
        return x
    else:
        raise ValueError(f"Expected a number, got {x} of type {type(x)}")


def pickle_hash(obj: Any) -> str:
    """
    Hashes the given object using pickle and md5.
    """
    pickled = pickle.dumps(obj)
    hashed = hashlib.md5(pickled).hexdigest()
    return hashed


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

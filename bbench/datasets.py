"""Download, split, and optionally standardize Bayesian benchmark datasets from HuggingFace."""

import hashlib
from dataclasses import dataclass

import numpy as np
from datasets import load_dataset as hf_load_dataset

from bbench.registry import hf_repo_id, __meta_by_name


@dataclass
class DataSplit:
    input_train: np.ndarray
    target_train: np.ndarray
    input_test: np.ndarray
    target_test: np.ndarray
    input_mean: np.ndarray | None = None
    input_std: np.ndarray | None = None
    target_mean: np.ndarray | None = None
    target_std: np.ndarray | None = None


def _md5(X: np.ndarray, Y: np.ndarray) -> str:
    h = hashlib.md5()
    h.update(np.ascontiguousarray(X, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(Y, dtype=np.float64).tobytes())
    return h.hexdigest()


def _normalize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(X, axis=0, keepdims=True)
    std = np.std(X, axis=0, keepdims=True) + 1e-6
    return (X - mean) / std, mean, std


def load_dataset(
    name: str,
    seed: int | np.random.Generator = 0,
    train_fraction: float = 0.9,
    standardize: bool = True,
) -> DataSplit:
    """Load a dataset, split into train/test, and optionally standardize.

    Args:
        name: Dataset name (use ``bbench.all_datasets()`` to see options).
        seed: An ``int`` seed or a ``numpy.random.Generator`` for shuffling.
        train_fraction: Fraction of data used for training (default 0.9).
        standardize: Whether to z-score standardize features and targets.
            When True, features are always standardized; targets are
            standardized only for non-classification datasets.
            When False, raw data is returned as-is.
    """
    if name not in __meta_by_name():
        raise KeyError(
            f"Unknown dataset '{name}'. Use bbench.all_datasets() to list available names."
        )
    meta = __meta_by_name()[name]

    ds = hf_load_dataset(hf_repo_id(), name, split="train")
    X = np.array(ds["features"], dtype=np.float64)
    Y = np.array(ds["target"], dtype=np.float64)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    if meta.md5 is not None and _md5(X, Y) != meta.md5:
        raise ValueError(
            f"Dataset '{name}': checksum mismatch. "
            "Data may be corrupted or the upstream dataset has changed."
        )

    rng = seed if isinstance(seed, np.random.Generator) else np.random.Generator(np.random.PCG64(seed))
    ind = np.arange(len(X))
    rng.shuffle(ind)

    n = int(len(X) * train_fraction)
    input_train, input_test = X[ind[:n]], X[ind[n:]]
    target_train, target_test = Y[ind[:n]], Y[ind[n:]]

    if not standardize:
        return DataSplit(input_train, target_train, input_test, target_test)

    input_train, input_mean, input_std = _normalize(input_train)
    input_test = (input_test - input_mean) / input_std

    if meta.category == "classification":
        return DataSplit(
            input_train, target_train, input_test, target_test,
            input_mean, input_std,
        )

    target_train, target_mean, target_std = _normalize(target_train)
    target_test = (target_test - target_mean) / target_std
    return DataSplit(
        input_train, target_train, input_test, target_test,
        input_mean, input_std, target_mean, target_std,
    )


def list_datasets(category: str | None = None) -> list[str]:
    """List available dataset names, optionally filtered by category.

    Args:
        category: ``"regression"``, ``"classification"``, ``"reinforcement"``, or ``None`` for all.
    """
    from bbench.registry import (
        all_datasets,
        classification_datasets,
        regression_datasets,
        reinforcement_datasets,
    )

    if category is None:
        return all_datasets()
    dispatch = {
        "regression": regression_datasets,
        "classification": classification_datasets,
        "reinforcement": reinforcement_datasets,
    }
    if category not in dispatch:
        raise ValueError(
            f"Unknown category '{category}'. Use 'regression', 'classification', or 'reinforcement'."
        )
    return dispatch[category]()

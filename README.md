# Bayesian Bechmarks (`bbench`)

Thin wrapper around [HuggingFace Datasets](https://huggingface.co/docs/datasets) for the
[Bayesian Benchmarks](https://huggingface.co/datasets/OccaMLab/bayesian-benchmarks) collection.

All data lives on HuggingFace Hub. This library adds shuffled train/test splitting and optional z-score standardization on top. If you need more control, use `datasets` directly (see [below](#using-datasets-directly)).

## Installation

With [uv](https://docs.astral.sh/uv/) (recommended):

```bash
uv pip install -e /path/to/bbench
```

Or in a uv-managed project, add it as a dependency:

```bash
uv add /path/to/bbench
```

With pip:

```bash
pip install -e /path/to/bbench
```

## Quick start

```python
import bbench

# Browse available datasets by category
bbench.regression_datasets()       # ['3droad', 'airfoil', 'autompg', ...]
bbench.classification_datasets()   # ['abalone', 'acute-inflammation', ...]
bbench.reinforcement_datasets()    # ['Ant-v2', 'HalfCheetah-v2', ...]

# Load a dataset (seed=0, 90/10 train/test, standardized)
data = bbench.load_dataset("boston")
data.input_train.shape   # (455, 13)
data.target_train.shape  # (455, 1)

# Custom seed
data = bbench.load_dataset("boston", seed=42)

# Pass a numpy Generator directly
import numpy as np
rng = np.random.Generator(np.random.PCG64(42))
data = bbench.load_dataset("boston", seed=rng)

# Change train/test fraction
data = bbench.load_dataset("boston", train_fraction=0.8)

# Disable standardization to get raw data
data = bbench.load_dataset("boston", standardize=False)
```

### Standardization

Enabled by default (`standardize=True`). Features are always z-score standardized. Targets are standardized for regression and reinforcement datasets but left as raw class labels for classification. Stats are computed on the training set and applied to the test set.

To recover original-scale regression predictions:

```python
y_original = data.target_std * y_predicted + data.target_mean
```

Pass `standardize=False` to skip all normalization and get raw arrays.

## Using datasets directly

```python
from datasets import load_dataset
import numpy as np

ds = load_dataset("OccaMLab/bayesian-benchmarks", "boston", split="train")
X = np.array(ds["features"])
Y = np.array(ds["target"])
```

Each HF config has a single `"train"` split with columns `features` (list of floats) and `target` (list of floats).

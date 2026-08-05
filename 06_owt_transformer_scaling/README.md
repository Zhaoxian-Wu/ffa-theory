# OpenWebText Transformer scaling

**Paper output:** Table 2, OpenWebText Transformer perplexity under Chinchilla-style budgets.

This directory is an installable Python package.
Install it from the repository root with `python -m pip install -e 06_owt_transformer_scaling`.
The public CLI is:

```bash
python -m local_learning_nanogpt.experiments.cli --help
```

The package source is in `local_learning_nanogpt/`, including CLI, scaling runner, local-learning algorithms, and data-preparation utilities.

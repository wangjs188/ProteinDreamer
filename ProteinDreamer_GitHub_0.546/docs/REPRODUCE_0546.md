# Run notes

Setup and commands: [README](../README.md).

## Quick GPU check

After setup, run the included 922-variant example:

```bash
python script/proteingym/run_example_0546.py
```

Expected: `pass_within_tolerance: true` in `result/example_0546/comparison.json`. Tested on an NVIDIA A40 (48 GB); the full 217-assay training and inference runs have not been revalidated end to end.

## Repeat a run

Use new output directories when repeating inference:

```bash
SCORES_ROOT=data/proteingym_local/scores_repeat EVAL_OUT=result/eval_repeat \
  bash script/proteingym/reproduce_best_0546.sh
```

For repeat training, use new `OUT_ROOT`, `SCORES_ROOT` and `EVAL_OUT` paths.

## Inputs and evaluation

The 0.546 configuration uses the supplied folds, caches and default scoring parameters. [Cache metadata](../data/proteingym_local/CACHE_PROVENANCE.json): 214 cache pairs and three historical no-cache cases.

The folds contain 217 assays from 186 UniProt IDs, with 20 IDs shared across validation folds. Hyperparameter selection was not fully nested.

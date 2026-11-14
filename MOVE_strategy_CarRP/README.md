# MOVE strategy: CarRP site selection

Minimal code and input scores for reproducing CarRP site filtering described in manuscript Section 4.4.

## Run

From this directory, run:

```bash
python select_sites.py
```

Requires Python 3; no additional packages are needed.

## Files

- `select_sites.py`: filtering rules and calculations.
- `data/structures/`: four ligand-bound CarRP PDBs used as FoldX inputs.
- `data/sites.csv`: 52 starting sites.
- `data/substitutions.csv`: input scores for 988 substitutions (19 per site).
- `results/`: generated stage counts, site decisions, selected sites and selected substitutions.

## Results

| Stage | Sites | Substitutions |
|---|---:|---:|
| Structural candidates | 52 | 988 |
| After FoldX | 34 | 646 |
| After risk filtering | 33 | 408 |
| Selected sites | 5 | 61 |

Selected sites: Y27, T31, R98, Q266 and G312.

The 61 substitutions are computational candidates across these five sites.

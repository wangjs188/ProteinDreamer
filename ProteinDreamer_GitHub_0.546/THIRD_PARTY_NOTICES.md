# Third-party notices

This release uses or interoperates with the following third-party projects and assets, which are not redistributed here unless explicitly stated:

- SaProt model code and pretrained checkpoints from `westlake-repl` on Hugging Face.  Users must follow the licenses and terms stated on the corresponding model cards.
- ProteinGym code, datasets, structures, MSA files, and sequence weights. The frozen 217-row reference metadata and small example inputs are included, including one complete assay with derived MSA/SiteRM caches and reference predictions under `data/example_0546/`; the archived benchmark MSA/SiteRM caches and metadata are also included with coverage recorded in `data/proteingym_local/CACHE_PROVENANCE.json`. The metric-reproduction inputs under `data/metric_0546/` include experimental measurements joined to the archived ProteinDreamer predictions, with source hashes and an exact join procedure recorded in `PROVENANCE.json` and `script/release/export_metric_inputs.py`. Other ProteinGym datasets are external. Users must follow the upstream licenses and dataset terms.
- Foldseek, used to derive 3Di structure tokens.
- CherryML/SiteRM, used when rebuilding the phylogenetic teacher cache.

The root `LICENSE` covers ProteinDreamer code as stated there. Users remain
responsible for the separate terms of externally downloaded SaProt, ProteinGym,
Foldseek and CherryML assets.

The two historical ProteinGym evaluation scripts under `external/ProteinGym-main/proteingym/` are redistributed unchanged under the included upstream MIT [license](external/ProteinGym-main/LICENSE). Their content hashes are recorded in `external/ProteinGym-main/SNAPSHOT.json`.

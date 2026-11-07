# ProteinDreamer

ProteinDreamer predicts mutation effects with SaProt. Reported ProteinGym average Spearman: **0.546** across **217 assays**, using supervised five-fold evaluation.

## 1. Setup

Use Linux and an NVIDIA GPU. Run all commands from the repository root:

```bash
conda env create -f environment.yaml
conda activate proteindreamer0546
bash script/setup/download_saprot_weights.sh
bash script/setup/download_proteingym_dataset.sh data/proteingym_downloads
bash script/setup/prepare_proteingym_saprot_dataset.sh
```

ProteinDreamer checkpoints, fixed folds, historical structures and MSA/SiteRM caches are included. Setup uses the structures from the reported run.

## 2. Train

```bash
bash script/train_0546.sh
```

This trains both SaProt-based branches across five folds, then runs inference and evaluation. Checkpoints are saved to `model/retrained_0546/`; evaluation results to `result/retrained_0546/`.

## 3. Inference

```bash
bash script/proteingym/reproduce_best_0546.sh
```

This uses the released checkpoints; retraining is optional. Expected `Average_Spearman` for `ProteinDreamer_final_lite_b130`: **0.546**. Results:

```text
result/proteingym_eval_best_0546/Spearman/Summary_performance_DMS_substitutions_Spearman.csv
```

Fresh training runs may give different numerical results. See [run notes](docs/REPRODUCE_0546.md) and [third-party notices](THIRD_PARTY_NOTICES.md).

## Citation

Jinsheng Wang, Haoran Hong, Meng Zha, Jinying Guo, Qingyan Li, Zhe Sun, and Xueli Zhang (2026). *Action-conditioned protein world modeling and multidimensional mutation-site prioritization for rational enzyme engineering*. Manuscript.

## Contact

Tianjin Institute of Industrial Biotechnology, Chinese Academy of Sciences.

- Zhe Sun: [sunzhe@tib.cas.cn](mailto:sunzhe@tib.cas.cn)

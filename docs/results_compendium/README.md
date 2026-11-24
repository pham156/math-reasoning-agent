# Completed-Results Atlas

Snapshot dated November 24, 2026: 28 pages, 59 comparison tables and three multi-panel figures, with 711 archived records. Includes completed negative results, not only accuracy improvements.

- [PDF](completed_results.pdf)
- [Modular LaTeX](main.tex)
- [Standalone LaTeX](all_results.tex)
- [Overleaf archive](results_overleaf.zip)
- [Numerical ledger and provenance](results_ledger.json)

The four opening pages specify architecture, datasets/splits, objectives, training settings and evaluation protocols. Results cover GRPO, Dr. GRPO, DAPO-style and GSPO; SFT/RFT controls; optimizer, reward and exploration ablations; inference-time selection; and transfer benchmarks. Supplements include learning-signal statistics, saved restart checks, category/level analyses, and extraction diagnostics.

All referenced figures and TeX inputs are included. Compile either entry point with XeLaTeX; select `main.tex` when importing the archive into Overleaf. Compiling the included sources does not need raw training outputs. Regeneration through `scripts/paper/build_compendium.sh` requires the original evidence and isolated runtimes and rejects incomplete or mismatched cohorts.

The ledger preserves source hashes and historical local paths for provenance, not runnable/downloadable checkpoint links. Older 7B results whose source artifacts were unavailable are not reconstructed from informal summaries. This atlas is broader and newer than the focused midterm/final/poster snapshots. DAPO-style is an objective variant without dynamic sampling, not a full DAPO reproduction. Reconstructed intermediate checkpoints have rollout/metric-prefix evidence, not unavailable original-weight hash comparisons.

# Final Report Draft

IEEE conference-style draft, snapshot dated November 15, 2026. Focuses on GRPO and Dr. GRPO, no-RL/SFT controls, training and reward ablations, longer training, exploration, and test-time selection.

- [Draft PDF](final_report.pdf)
- [LaTeX](main.tex)
- [Overleaf archive](final_overleaf.zip)

Compile `main.tex` with the standard IEEEtran class. All 15 referenced table inputs and two dense-training vector figures are included in `generated/` and `figures/`; raw scratch outputs are not required for compilation. The 18-file Overleaf archive contains the same manuscript assets.

The draft is a dated, deliberately narrower snapshot than the [results atlas](../results_compendium/completed_results.pdf). In particular, its frozen follow-up export omits a consensus cohort recovered later. Do not treat that earlier omission as evidence that the later validated cohort failed.

The bundled sources compile to ten pages. Existing IEEE float/paragraph underfull-box spacing warnings remain; there are no missing assets or unresolved references.

`scripts/paper/build_final.sh` regenerates evidence and figures before compilation and therefore requires the original saved studies, isolated runtimes and Tectonic. For editing the included manuscript without that evidence, compile the supplied LaTeX directly instead.

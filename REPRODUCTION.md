# TabularMark reproduction

This repository tracks a reproducible rerun of the experiments in
"TabularMark: Watermarking Tabular Datasets for Machine Learning."

## Execution policy

- Code and reports are maintained locally and committed to GitHub.
- Computational experiments run in the dedicated `tabularmark-repro` Docker
  container on `hustserver65`. The host project path is
  `/data/d2023-fsy/tabularmark-reproduction`, mounted at
  `/workspace/tabularmark-reproduction` in the container.
- The author-provided dataset archive is transferred directly to the server
  and is not committed to GitHub.
- Each meaningful experiment iteration is committed and pushed before the
  next iteration starts.

## Completed coverage

1. Detectability and false-positive behavior (Table 2 and Figure 6).
2. Non-intrusiveness on classification and regression tasks (Tables 3-4).
3. Robustness to alteration, insertion, and deletion attacks (Tables 6-11).
4. Hyperparameter trade-offs (Figures 8-10 and Tables 20 and 22).
5. Paper/artifact/reproduction comparison in `REPRODUCTION_REPORT.md`.

The Zoo, Adult, and Hospital experiments are not included because their data
are not present in the author-provided archive. Comparison baselines that need
external implementations are also outside the self-contained artifact scope.

# TabularMark reproduction

This repository tracks a reproducible rerun of the experiments in
"TabularMark: Watermarking Tabular Datasets for Machine Learning."

## Execution policy

- Code and reports are maintained locally and committed to GitHub.
- Computational experiments run on `hustserver65` under
  `/data/d2023-fsy/tabularmark-reproduction`.
- The author-provided dataset archive is transferred directly to the server
  and is not committed to GitHub.
- Each meaningful experiment iteration is committed and pushed before the
  next iteration starts.

## Planned coverage

1. Detectability and false-positive behavior (Table 2 and Figure 6).
2. Non-intrusiveness on classification and regression tasks (Tables 3-5).
3. Robustness to alteration, insertion, and deletion attacks (Tables 6-11).
4. Hyperparameter trade-offs (Figures 8-10 and Table 22).
5. A machine-readable comparison between reproduced and reported results.


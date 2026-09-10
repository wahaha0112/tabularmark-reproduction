# TabularMark

This fork contains a server-verified reproduction of all numbered experiments
in the CCS 2024 paper. Start with:

- [`REPRODUCTION_REPORT.md`](REPRODUCTION_REPORT.md): Chinese final report,
  paper/result comparison, new findings, and conclusions.
- [`REPRODUCTION.md`](REPRODUCTION.md): environment, protocol substitutions,
  and execution commands.
- [`reproduce.py`](reproduce.py): unified deterministic runner.
- [`outputs/`](outputs/): CSV tables, PNG figures, and run metadata from
  `hustserver65`.

The complete suite runs inside the `tabularmark-repro` Docker container with:

```bash
.conda-env/bin/python reproduce.py --stage all --xgb-device gpu
```

The latest complete dual-A100 run took 130.99 seconds. Table 12, Table 15, and
Table 27 use explicitly labelled reconstructed protocols because the upstream
artifact does not publish enough information to execute the original protocols.

## Upstream description

This repository implements experiments for TabularMark. The idea is to perturb a small proportion of cells in a tabular dataset to embed watermarks.

## Dataset
The datasets used in the experiments can be downloaded from [this link](https://drive.google.com/file/d/10efT2gKtR8BjDOwYkQwDnirQtsA5wIxW/view?usp=sharing). After downloading `datasets.zip`, unzip it and place the `dataset` folder in the TabularMark directory.

## Usage
All the experimental code is located in the `/experiments` directory. Each experiment is contained in its respective folder. To run an experiment, simply navigate to the appropriate directory and execute the scripts. 

For example, to run `experiment1`:
```bash
cd experiments/experiment1
python script1.py
```

There are also some experimental codes in Jupyter Notebook scripts. You can run each cell according to the annotations to obtain the results.

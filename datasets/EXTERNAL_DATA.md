# External dataset provenance

The original artifact archive omitted several datasets used only by the extended
experiments. The following files were retrieved on 2026-09-10 and are kept here
so the offline server run remains reproducible.

| Local path | Upstream source | SHA-256 |
|---|---|---|
| `zoo/zoo.data` | UCI Zoo dataset, archive id 111 | `cddc71c26ab9bc82795b8f4ff114cade41885d92720c6af29ffb69bcf73f0315` |
| `zoo/zoo.names` | UCI Zoo dataset, archive id 111 | `db3dd334beb643c9e14b75603360ea5edcac90a6ac19302e1896039f0c20d428` |
| `holoclean/Adult1100.csv` | HoloClean `testdata/Adult1100.csv` | `30816ea6cc51ee354b3647efffbe92d72882b90ae6cc9ddd5dc64d29f2f817b0` |
| `holoclean/adult_constraints.txt` | HoloClean `testdata/adult_constraints.txt` | `fe02c4d7c525bd79e1d9d7187a09e0a543151ae7d0138b1661016d8385349d65` |
| `holoclean/hospital.csv` | HoloClean `testdata/hospital.csv` | `bbb2f60e9e7bbda68b1115b3bbb9a0d70587a9d33384a2373e4d447789fd619a` |
| `holoclean/hospital_constraints.txt` | HoloClean `testdata/hospital_constraints.txt` | `0eb05d12dfbb7bcd213ff086c04c971314a68fbb85ba4532e763e399e0085825` |

Upstream URLs:

- <https://archive.ics.uci.edu/dataset/111/zoo>
- <https://github.com/HoloClean/holoclean/tree/master/testdata>

The Iris dataset is loaded from the copy bundled with scikit-learn, so it does
not require a separate network download.

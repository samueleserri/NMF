# NMF (Non-negative Matrix Factorization) —  Code

Main Repo for my bachelor thesis on Non-negative matrix factorization.

## Contents
- `src/` — core code.
    - `src/nmf` - package with algorithms for non negative matrix factorization.
        - `NMF.py` — base NMF class.
        - `SepNMF.py` — separable NMF (SNPA).
        - `NonNegMatrix.py` — extension of np.array for NonNegative matrices.
    - `src/utils/` — utility helpers.
        - `Data_Gen.py` — synthetic data generator.
- `src/examples/` — contains examples of NMF application on real data.
  - `example_topics_extarction.py`
- `data/` - all datasets used
  -`CBCL.csv` — face dataset used in https://www.nature.com/articles/44565

## Requirements
- Python 3.8+
- numpy
- scipy
- scikit-learn
- matplotlib
- pandas
 
Install editable from source with pip (uses pyproject.toml):

  ```bash
  python -m pip install -e .
  ```
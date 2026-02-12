# NMF (Non-negative Matrix Factorization) —  Code

This code was developed as part of  my bachelor thesis work at Trieste University. 
It contains the implementation of algorithms for solving the Non-Negative matrix factorization (NMF) problem.

The NMF problem consists in: given a non-negative matrix $V \in \mathbb{R}^{m \times n}$ and a factorization rank $r$, find to non-negative matrices $W \in \mathbb{R}^{m \times r}$ and $W \in \mathbb{R}^{r \times n}$ such that their product approximates  the original matrix $V$ as closely as possible.  
This is defined in terms of a constrained optimization problem:
```math
\min_{W \in \mathbb{R}^{m \times r}, H \in \mathbb{R}^{r \times n}} D(V, WH) \\
\text{subject to} \\
W \geq 0, H \geq 0
```
And it is a widely used technics in data analysis whenever dealing with inherently non-negative data.
## Repository Structure

- **NMF/**
  - `src/`
    - `nmf/`
      - `__init__.py`
      - `NMF.py` — base NMF class
      - `RegularizedNMF.py` — NMF subclass with regularization
      - `SparseNMF.py` — NMF subclass for sparse NMF models
      - `NonNegMatrix.py` — non‑negative `numpy.ndarray`
    - `utils/`
      - `__init__.py`
      - `beta_divergence.py` — compute beta‑divergence
      - `display.py` — helper to display images
      - `measure_sparsity.py` — measure matrix sparsity
    - `examples/`
      - `minimal_examples/`
        - `minimal_example.py` # basic usage of `NMF`
        - `example_regularizer.py` # use `RegularizedNMF` with a custom regulariser
      - `example_feature_extraction.py` # feature extraction from the CBCL dataset
      - `example_feature_extraction_olivetti_dataset.py` # feature extraction from the Olivetti dataset
      - `example_topics_extraction.py` # topic extraction from the 20‑newsgroups dataset
      - `example_top_30.py` # topic modelling with the top‑30 news dataset
  - `data/`
    - `CBCL.csv` # face dataset used by Lee 1999
    - `Swimmer.csv` # 220 × 256‑pixel swimmer images
    - `tdt2_top30.mat` # top‑30 news dataset for topic modelling
  - `tests/`
    - `beta_divergence_plot.py` # script to plot the beta‑divergence
  - `README.md`


## Getting Started

To get started, clone the repository and navigate to the project directory:

```bash
git clone https://github.com/samueleserri/NMF.git 
cd NMF
```
or download the ZIP file and extract it.
Then, navigate to the project directory:
```bash
cd NMF
``` 

### Install in a virtual environment

It is recommended to install this package in a virtual environment to avoid conflicts with other packages.
To do this, create a virtual environment using venv1; open your terminal and (inside the NMF directory) run:
```bash
python -m venv venv
```

Then activate your virtual environment:
```bash
source venv/bin/activate  # On macOS/Linux
```

Finally, inside the virtual environment, install the package in editable mode using pip:
```bash
pip install -e .
```
This will install all required dependencies as well; the command uses the pyproject.toml file.
To verify the installation, you can run:
```bash
pip list
```
to see the installed packages. You should see a package named NMF and its dependencies.

## Requirements
- Python 3.8+
- numpy
- scipy
- scikit-learn
- matplotlib
- pandas

## Example usage
### Basic usage

First import the NMF class from the nmf package:
```bash
from nmf import NMF
import numpy as np
```
Then create a non-negative data matrix, in this example we generate a random matrix sampled from a uniform distribution:
```bash
V = np.random.rand(100, 20) @ np.random.rand(20, 100)
```
Then instantiate the NMF class with the data matrix and desired rank, in this case we set rank=20 because we constructed V as the product of two matrices of rank at most 20:
```bash
model = NMF(V, rank=20, max_iter=10000, tol=1e-4, T=10)
```
To fit the model we simply call the fit method with the desired solver, in this case we use the Multiplicative Updates (MU) algorithm and the Kullback-Leibler divergence (beta=1):
```bash
model.fit(solver="beta_MU", beta=1)
print(f"Final error: {model.get_final_error()}")
```
You should see the output:

<!-- Black‑terminal style -->
<pre style="
    background:#000;         
   color:#fff;               
   font-size:12.5px;                  
    padding:1em;
    border-radius:4px;
    overflow:auto;
    font-family:monospace;
">
Fitting with beta_MU algorithm

value of beta: 1

Fit completed in 3.8579 s, iterations: 10000, avg time/iter: 3.8579e-04 s

Max iter reached: you may try to increase the value

Final error: 1.33448729119308e-05
</pre>




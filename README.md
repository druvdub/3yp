# 3YP

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Exploratory project for NDR with SNN

Dataset available on Huggingface: (https://huggingface.co/datasets/ddosdub/CSECICIDS2018)[https://huggingface.co/datasets/ddosdub/CSECICIDS2018]

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Model checkpoints
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│                      <- Notebooks are for exploration and experimentation: grouped by type of architecture and experiments
│
├── pyproject.toml     <- Project configuration file with package metadata for
│                         3yp and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── 3yp   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes 3yp a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── models                  <- all available model classes
        └── __init__.py
```

---

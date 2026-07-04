# notebooks/

Notebooks **orchestrate**; all reusable logic lives in `src/wildfire_rl/`. Outputs are
stripped on commit (`nbstripout` via pre-commit) so diffs stay reviewable and the repo stays
small (the original notebooks embedded multi-MB base64 plots).

```
notebooks/
├── 01_quickstart_demo.ipynb   # minimal end-to-end demo using the installed package
└── legacy/                     # original research notebooks, preserved for provenance
    ├── California/             #   data pipelines + training + transfer
    ├── Saudi/                  #   pipelines + training + MARL + ablations + stats
    └── 10_final_analysis_and_figures.ipynb
```

The `legacy/` notebooks are the historical record of how the project was developed. They
contain hardcoded Colab paths and duplicated environment code and are **not** the supported
execution path — use the CLI / scripts (`wildfire-rl ...`, `python scripts/*.py`) instead.
Their logic has been refactored into `src/wildfire_rl/` (see `docs/architecture.md` for the
notebook→module mapping).

Run notebooks against a real kernel after `pip install -e .`. Parameterize heavy runs with
[papermill](https://papermill.readthedocs.io/) + the `configs/` YAML rather than editing cells.

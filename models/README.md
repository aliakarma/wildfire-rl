# models/

Trained PPO checkpoints live here at runtime but are **not tracked in git** (each is
~200 MB; the full set is ~2.2 GB). They are distributed via the Hugging Face Hub and
verified against `results/models_manifest.json`.

```bash
python scripts/fetch_models.py --repo aliakarma/wildfire-rl-ppo \
    --verify results/models_manifest.json
```

Naming convention: `ppo_<region>_<grid>_seed_<seed>.zip`
(e.g. `ppo_saudi_32_seed_0.zip`). MARL: `ppo_marl_<region>_<N>agents_seed_<seed>.zip`.

Everything in this directory except this README is git-ignored — see `.gitignore`.

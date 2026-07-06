# Pre-Markov checkpoints (DEPRECATED)

All checkpoints here were trained on the OLD 7-channel observation (no agent-position
channel). After the Phase 3 Markov fix the environment emits an 8-channel observation, so
the CNN's first conv layer (in_channels=8) is INCOMPATIBLE with these weights (in_channels=7).
They will fail to load into the current env and MUST NOT be used for reported results.
Retained for provenance only. Fresh checkpoints are produced in Phase 9.

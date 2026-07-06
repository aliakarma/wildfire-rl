"""wildfire_marl — decentralized multi-agent RL for wildfire suppression on validated
fire physics (Cell2Fire), with critical-infrastructure protection and cross-regional
transfer (Saudi Arabia <-> California).

V2 re-founding of the project (see ``REMEDIATION_PLAN_V2.md``). The reusable evaluation
and reproducibility core is ported from the frozen V1 prototype (``legacy_v1/``); the
toy simulator, oracle heuristics, and stubbed controller were retired and are NOT part
of this package. Nothing under ``legacy_v1/`` is imported here.
"""

from __future__ import annotations

__version__ = "2.0.0a0"

__all__ = ["__version__"]

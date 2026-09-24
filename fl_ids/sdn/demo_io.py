"""Fixed file paths the Phase B orchestrator and the root-run topology script share.

`fl_ids/sdn/topology.py` runs as root under a sudoers rule that allows
only the exact command `/usr/bin/python3 <path>/topology.py`, with no
arguments, so its parameters come from a file at a fixed path, and it
reports back through another. Kept in a module with no heavy imports, so
the orchestrator (a normal-user process without Mininet) and the topology
script (root, with Mininet) can both import it.
"""

from __future__ import annotations

from pathlib import Path

DEMO_CONFIG_PATH = Path("/tmp/fl_ids_topology_demo_config.json")
DEMO_RESULT_PATH = Path("/tmp/fl_ids_live_demo_result.json")

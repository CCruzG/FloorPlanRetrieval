"""Backend for the E2E gate (2026-11-04): the product Flask app on its own port,
with CP-SAT set to the gate's settings (1 worker, deterministic time).
No product file is changed: spec.build_options is wrapped by tools/gate_common.
Usage: run_backend.py --port 5012 [--workers 1] [--det-time 1.0]
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=5012)
ap.add_argument("--workers", type=int, default=1)
ap.add_argument("--det-time", type=float, default=1.0)
a = ap.parse_args()

from apps.api import server as server_mod  # noqa: E402
from packages.floorplan_core.index_store import build_index_store_from_npz  # noqa: E402
from tools.gate_common import patch_solver  # noqa: E402

patch_solver(a.workers, a.det_time)
server_mod.INDEX_STORE = build_index_store_from_npz(
    npz_path=str(REPO / "data/index/rplan_index_v5.npz"),
    data_dir=str(server_mod.RPLAN_DATASET_DIR), logger=server_mod.logger)
print(f"gate backend ready on {a.port}", flush=True)
server_mod.app.run(host="127.0.0.1", port=a.port, debug=False, threaded=True)

"""NeuralFn CUDA Tile example."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "cli"))

from cli.nfn_impl import main


raise SystemExit(main(["kernels", "bench", "--iterations", "200"]))

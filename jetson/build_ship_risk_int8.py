#!/usr/bin/env python3
from pathlib import Path

import build_int8


ROOT = Path(__file__).resolve().parent

# Reuse the verified calibration implementation, but keep the new model's
# artifacts separate from the previous deployment.
build_int8.ONNX_PATH = ROOT / "ship_risk_model.onnx"
build_int8.CACHE_PATH = ROOT / "ship_risk_int8.cache"
build_int8.ENGINE_PATH = ROOT / "ship_risk_int8.engine"


if __name__ == "__main__":
    build_int8.main()

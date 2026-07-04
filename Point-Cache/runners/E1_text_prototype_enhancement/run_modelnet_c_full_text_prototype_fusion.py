#!/usr/bin/env python
"""Official E1 entry point for ULIP x ModelNet-C full text prototype fusion."""

import sys
from pathlib import Path

POINT_CACHE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(POINT_CACHE_ROOT))

from runners.E1_text_prototype_enhancement.run_e1_ulip_modelnetc_s2_zs_prompt_ablation import main


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""02_16_1 entry point for 02_9_2 cache branch diagnostics.

The implementation intentionally reuses the existing E7-B3-Diag-02_9_2 runner
so the diagnostic code has a single maintenance surface.
"""

from pathlib import Path
import sys

POINT_CACHE_ROOT = Path(__file__).resolve().parents[2]
if str(POINT_CACHE_ROOT) not in sys.path:
    sys.path.insert(0, str(POINT_CACHE_ROOT))

from runners.E7_entropy_energy_alignment_multicache.run_e7_b3_diag_0292_ulip_modelnetc_s2_cache_branch_diagnostics import main


if __name__ == "__main__":
    main()

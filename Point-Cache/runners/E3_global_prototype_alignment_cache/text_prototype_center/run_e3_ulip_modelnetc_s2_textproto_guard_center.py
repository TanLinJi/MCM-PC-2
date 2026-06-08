#!/usr/bin/env python
"""
Minimal wrapper for E3-V2-TextProto-Guard-C.

No copied runner logic here. This wrapper reuses the existing E3 ULIP evaluation
CLI and swaps only the hierarchical cache function.
"""

import sys
from pathlib import Path

POINT_CACHE_ROOT = Path(__file__).resolve().parents[3]
if str(POINT_CACHE_ROOT) not in sys.path:
    sys.path.insert(0, str(POINT_CACHE_ROOT))

from runners.E3_global_prototype_alignment_cache import (
    run_e3_ulip_modelnetc_s2_parallel_gpa_entropy_gpa_union_center as base_runner,
)
from runners.E3_global_prototype_alignment_cache.text_prototype_center.model_with_hierarchical_caches_textproto_guard_center import (
    run_test_tda as run_hierarchical_cache,
)

base_runner.run_hierarchical_cache = run_hierarchical_cache

if __name__ == "__main__":
    base_runner.main()

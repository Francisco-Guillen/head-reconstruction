import time
from contextlib import contextmanager

_timings: dict = {}

@contextmanager
def timer(stage_name: str):
    start = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start
        _timings[stage_name] = _timings.get(stage_name, 0) + elapsed
        print(f"[TIMER] {stage_name}: {elapsed:.2f}s")

def get_timings() -> dict:
    return _timings

def print_summary():
    print("\n=== Timing Summary ===")
    total = 0.0
    for stage, t in _timings.items():
        print(f"  {stage:<30} {t:.2f}s")
        total += t
    print(f"  {'TOTAL':<30} {total:.2f}s")
    print("======================\n")

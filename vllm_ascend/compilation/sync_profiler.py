# SPDX-License-Identifier: Apache-2.0
"""Env-gated per-site synchronize() profiler for the decode path.

VLLM_ASCEND_SYNC_PROFILE=1 wraps the hot synchronize sites with wall-clock
timing and logs a per-site count/ms table every ``_LOG_EVERY`` calls, so the
marginal ~3ms/iteration of host-side sync (measured by S/A/U diff of
aclrtSynchronizeEvent) can be attributed to exact call sites. A host trace
would do this too, but the with-stack profrun data lost its host stream to a
SIGKILL; this is the deterministic fallback. Default (env unset) calls the
sync immediately with one extra attribute lookup.
"""
import os
import time
from collections import defaultdict

from vllm.logger import logger

_ENABLED = os.environ.get("VLLM_ASCEND_SYNC_PROFILE") == "1"
_LOG_EVERY = 256

_calls: dict[str, int] = defaultdict(int)
_seconds: dict[str, float] = defaultdict(float)


def _maybe_log() -> None:
    total = sum(_calls.values())
    if total > 0 and total % _LOG_EVERY == 0:
        table = ", ".join(
            f"{site}: n={n} t={_seconds[site] * 1e3:.1f}ms"
            for site, n in sorted(_calls.items(), key=lambda kv: -kv[1])
        )
        logger.info("SYNCPROF[%d] %s", total, table)


def record(site: str, elapsed_s: float) -> None:
    if not _ENABLED:
        return
    _calls[site] += 1
    _seconds[site] += elapsed_s
    _maybe_log()


def timed_sync(site: str, fn) -> None:
    """Run fn() (a synchronize call) and record its wall cost under site."""
    if not _ENABLED:
        fn()
        return
    t0 = time.perf_counter()
    fn()
    record(site, time.perf_counter() - t0)

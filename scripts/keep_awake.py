"""Keep a Windows machine from sleeping while an experiment script runs.

On 2026-09-29/30 two identical enumerations on the measurement machine took 4619 s and
11783 s, against about 1900 s for both in the earlier run, and the stage drivers asked the
system for nothing. Every script a stage runs calls keep_awake() first: it asks Windows to
keep the system running for as long as this process lives (ES_CONTINUOUS |
ES_SYSTEM_REQUIRED). It does not stop a sleep the user asks for, or one caused by closing a
laptop lid.

Everything is written to stderr. The stage driver decides the weight modes from whether
pick_stable_rho.py prints anything on stdout, so a line there would change which modes run.
On other systems nothing is done and nothing is printed.
"""

from __future__ import annotations

import sys

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def keep_awake(kernel32=None, platform: str = sys.platform, stream=None) -> bool:
    """Ask Windows not to sleep while this process runs; True if the request was accepted."""
    if not platform.startswith("win"):
        return False
    stream = stream if stream is not None else sys.stderr
    if kernel32 is None:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetThreadExecutionState.restype = ctypes.c_uint32
        kernel32.SetThreadExecutionState.argtypes = [ctypes.c_uint32]
    previous = kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    if not previous:
        print("[warning] Windows refused the keep-awake request: the machine may sleep during "
              "this run, and any wall-clock time it reports would include the sleep.",
              file=stream, flush=True)
        return False
    print("[awake] Windows will not sleep on its own while this script runs.",
          file=stream, flush=True)
    return True

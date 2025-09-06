"""Compatibility shim.
Importing `protrace_spn` was the old path.  The codebase has been reorganised
into a proper package under `src/protrace`.  This shim lets existing notebooks
or scripts that still do `import protrace_spn` keep working without changes.

In new code, please switch to::

    from protrace.core import ProTraceFingerprint, ProTraceConfig, ProTraceCLI
"""
import sys, os

# Ensure './src' is on the path so `import protrace` works when the project root
# is executed directly (e.g. `python examples/...`).
PROJECT_ROOT = os.path.dirname(__file__)
SRC_PATH = os.path.join(PROJECT_ROOT, 'src')
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from protrace.core import *  # re-export everything for backwards-compat

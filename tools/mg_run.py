"""python tools/mg_run.py -m duoprop.collective_eval ARGS...   or   python tools/mg_run.py SCRIPT.py ARGS...
Runs a module or script with the truth matrices spread over all visible GPUs (tools/multigpu.py)."""
import os
import runpy
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import multigpu  # noqa: E402,F401  (patches duoprop.reason.load_cache)

if sys.argv[1] == '-m':
    mod = sys.argv[2]
    sys.argv = [mod] + sys.argv[3:]
    runpy.run_module(mod, run_name='__main__', alter_sys=True)
else:
    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    runpy.run_path(script, run_name='__main__')

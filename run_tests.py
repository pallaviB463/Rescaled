"""Fallback test runner for machines without pytest:  python run_tests.py   (with pytest installed, just run `pytest`)."""
import sys, os, importlib, traceback, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src")); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tests"))
bad = 0
for f in sorted(glob.glob(os.path.join(os.path.dirname(__file__), "tests", "test_*.py"))):
    m = importlib.import_module(os.path.basename(f)[:-3])
    for n in sorted(x for x in dir(m) if x.startswith("test_")):
        try: getattr(m, n)(); print("PASS", m.__name__, n)
        except Exception: bad += 1; print("FAIL", m.__name__, n); traceback.print_exc(limit=3)
print("\nFAILURES:", bad); sys.exit(1 if bad else 0)

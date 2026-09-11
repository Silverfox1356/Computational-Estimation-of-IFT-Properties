"""
Test the IFT-curve CSV loader (core.ift_data.load_ift_curve).

Plain script (project convention — no pytest):
    conda run -n ift python tests/test_ift_csv.py

The loader feeds Step 8's fit directly, so a silently mis-read column
would bias D and k with no error.  Covers the accepted layouts and every
rejection branch.
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from core.ift_data import MAX_FILE_BYTES, IFTDataError, load_ift_curve

_passed = 0
_failed = 0


def check(name, cond):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  ok   {name}")
    else:
        _failed += 1
        print(f"  FAIL {name}")


def write(tmp, text, name, encoding="utf-8"):
    p = Path(tmp) / name
    p.write_text(text, encoding=encoding)
    return p


def expect_ok(name, path):
    try:
        return load_ift_curve(path)
    except IFTDataError as e:
        check(f"{name} -> accepted (got: {e})", False)
        return None


def expect_reject(name, path, must_contain=None):
    try:
        load_ift_curve(path)
        check(name + " -> rejected", False)
    except IFTDataError as e:
        ok = must_contain is None or must_contain in str(e)
        check(name + " -> rejected", ok)
        print(f"       msg: {e}")


def main():
    with tempfile.TemporaryDirectory() as tmp:
        print("── accepted layouts ──")
        d = expect_ok("header", write(tmp, "time_s,ift_mNm\n10,60\n20,59\n30,58.5\n", "a.csv"))
        check("header: values read", d is not None
              and np.allclose(d['t'], [10, 20, 30])
              and np.allclose(d['gamma'], [60, 59, 58.5]) and not d['warnings'])

        d = expect_ok("no header", write(tmp, "10,60\n20,59\n", "b.csv"))
        check("no header: first two columns", d is not None
              and np.allclose(d['t'], [10, 20]))

        d = expect_ok("comments + blanks",
                      write(tmp, "# digitised\n\n# t, ift\n10,60\n\n20,59\n", "c.csv"))
        check("comments + blanks: skipped", d is not None and len(d['t']) == 2)

        d = expect_ok("named columns reversed",
                      write(tmp, "IFT (mN/m),time (s)\n60,10\n59,20\n", "d.csv"))
        check("named columns reversed: mapped by name", d is not None
              and np.allclose(d['t'], [10, 20]) and np.allclose(d['gamma'], [60, 59]))

        d = expect_ok("extra columns", write(
            tmp, "pressure_MPa,t_s,gamma_mNm\n5.67,25.3,39.78\n5.67,72.0,39.11\n", "e.csv"))
        check("extra columns: time + IFT picked by name", d is not None
              and np.allclose(d['t'], [25.3, 72.0]) and np.allclose(d['gamma'], [39.78, 39.11]))

        d = expect_ok("UTF-8 BOM", write(tmp, "time,ift\n10,60\n20,59\n", "f.csv",
                                         encoding="utf-8-sig"))
        check("UTF-8 BOM: header still recognised", d is not None and not d['warnings'])

        d = expect_ok("unsorted", write(tmp, "30,58\n10,60\n20,59\n", "g.csv"))
        check("unsorted: sorted with a warning", d is not None
              and np.allclose(d['t'], [10, 20, 30]) and np.allclose(d['gamma'], [60, 59, 58])
              and any('sorted' in w for w in d['warnings']))

        d = expect_ok("unrecognised header", write(tmp, "a,b\n10,60\n20,59\n", "h.csv"))
        check("unrecognised header: first two columns + warning", d is not None
              and np.allclose(d['t'], [10, 20]) and d['warnings'])

        d = expect_ok("N/m values", write(tmp, "10,0.060\n20,0.059\n", "i.csv"))
        check("N/m values: unit warning", d is not None
              and any('N/m' in w for w in d['warnings']))

        print("── rejections ──")
        expect_reject("NaN IFT", write(tmp, "10,nan\n20,59\n", "r1.csv"), "NaN")
        expect_reject("Inf time", write(tmp, "inf,60\n20,59\n", "r2.csv"))
        expect_reject("zero IFT", write(tmp, "10,0\n20,59\n", "r3.csv"), "> 0")
        expect_reject("negative IFT", write(tmp, "10,-5\n20,59\n", "r4.csv"), "> 0")
        expect_reject("negative time", write(tmp, "-1,60\n20,59\n", "r5.csv"), "negative")
        expect_reject("formula not executed",
                      write(tmp, "10,=cmd|' /c calc'!A0\n20,59\n", "r6.csv"), "not a number")
        expect_reject("duplicate time", write(tmp, "10,60\n10,59\n20,58\n", "r7.csv"),
                      "duplicate")
        expect_reject("multi-series file (pressure column read as time)",
                      write(tmp, "0.49,19.5,65.8\n0.49,51.9,64.4\n1.88,43.1,58.4\n", "r8.csv"),
                      "one file per series")
        expect_reject("single point", write(tmp, "10,60\n", "r9.csv"), "at least")
        expect_reject("empty file", write(tmp, "", "r10.csv"), "at least")
        expect_reject("missing value", write(tmp, "10,\n20,59\n", "r11.csv"), "expected")
        expect_reject("text after data", write(tmp, "10,60\nfoo,bar\n", "r12.csv"),
                      "not a number")
        bad = Path(tmp) / "r13.csv"
        bad.write_bytes(b"10,60\n20,\xff59\n")
        expect_reject("not UTF-8", bad, "UTF-8")
        big = Path(tmp) / "r14.csv"
        big.write_text("10,60\n" * (MAX_FILE_BYTES // 6 + 10), encoding="utf-8")
        expect_reject("oversized file", big, "MB")
        expect_reject("missing file", Path(tmp) / "nope.csv", "cannot open")

    print(f"\n{_passed} passed, {_failed} failed")
    if _failed:
        sys.exit(1)


if __name__ == "__main__":
    main()

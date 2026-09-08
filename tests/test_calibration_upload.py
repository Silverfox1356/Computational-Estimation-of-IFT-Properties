"""
Test the calibration upload path: parse + validate + atomic commit.

Plain script (project convention — no pytest):
    conda run -n ift python tests/test_calibration_upload.py

Covers the ACID / all-or-nothing guarantees and every rejection branch.
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from core.calibration import (
    Calibration, CalibrationError,
    parse_calibration_template, install_calibration,
)

GOOD = """\
c_sat,
concentration_unit,mol/m3
concentration,ift
0,70.55
155.1,57.66
493.1,52.28
673,48.01
776.1,43.16
888.2,40.13
976,36.64
"""

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


def write(tmp, text, name="cal.csv"):
    p = Path(tmp) / name
    p.write_text(text, encoding="utf-8")
    return p


def expect_reject(name, text, tmp):
    p = write(tmp, text, name.replace(" ", "_") + ".csv")
    try:
        parse_calibration_template(p)
        check(name + " -> rejected", False)
    except CalibrationError as e:
        check(name + " -> rejected", True)
        print(f"       msg: {e}")


def main():
    tmp = tempfile.mkdtemp(prefix="cal_test_")

    # --- happy path ---------------------------------------------------
    p = write(tmp, GOOD)
    parsed = parse_calibration_template(p)
    check("valid file parses", len(parsed["c"]) == 7)
    check("blank c_sat -> endpoint (976)", parsed["c_sat"] == 976.0)
    check("no spurious warnings", parsed["warnings"] == [])
    check("unit read", parsed["unit"] == "mol/m3")

    # --- endpoint default builds a working Calibration ---------------
    canonical = Path(tmp) / "active.csv"
    calib, warns = install_calibration(p, canonical_path=canonical)
    check("install returns Calibration", isinstance(calib, Calibration))
    check("canonical file written", canonical.exists())
    check("gamma0 = first endpoint", calib.gamma0 == 70.55)
    check("gamma_inf = last endpoint", calib.gamma_inf == 36.64)
    # Cs=1 -> saturated -> lowest IFT; Cs=0 -> clean -> highest IFT.
    check("gamma(Cs=0) ~ 70.55", np.isclose(calib.gamma(0.0), 70.55))
    check("gamma(Cs=1) ~ 36.64", np.isclose(calib.gamma(1.0), 36.64))

    # --- atomic overwrite: second install replaces, never duplicates -
    before = canonical.read_text()
    p2 = write(tmp, GOOD.replace("36.64", "35.00"), "cal2.csv")
    install_calibration(p2, canonical_path=canonical)
    after = canonical.read_text()
    check("re-upload overwrote canonical", after != before and "35" in after)
    check("still exactly one canonical file",
          sum(1 for _ in Path(tmp).glob("active*.csv")) == 1)
    check("no leftover temp files",
          not any(x.suffix == ".tmp" for x in Path(tmp).iterdir()))

    # --- override + mismatch warning ---------------------------------
    over = GOOD.replace("c_sat,", "c_sat,976")            # matches endpoint
    parsed = parse_calibration_template(write(tmp, over, "over.csv"))
    check("matching override -> no warning", parsed["warnings"] == [])

    slip = GOOD.replace("c_sat,", "c_sat,0.976")          # 1000x unit slip
    parsed = parse_calibration_template(write(tmp, slip, "slip.csv"))
    check("unit-slip override -> warns", len(parsed["warnings"]) == 1)
    check("override honoured despite warning", parsed["c_sat"] == 0.976)

    # --- unsorted input is fixed with a warning ----------------------
    rev = "\n".join(GOOD.strip().splitlines()[:3] + [
        "976,36.64", "0,70.55", "493.1,52.28"]) + "\n"
    parsed = parse_calibration_template(write(tmp, rev, "rev.csv"))
    check("unsorted -> sorted + warned",
          parsed["c"][0] == 0 and len(parsed["warnings"]) == 1)

    # --- all-or-nothing: a failing upload leaves canonical untouched -
    good_text = canonical.read_text()
    try:
        install_calibration(write(tmp, "garbage\nnope\n", "bad.csv"),
                            canonical_path=canonical)
    except CalibrationError:
        pass
    check("failed upload did NOT touch canonical",
          canonical.read_text() == good_text)

    # --- rejection branches ------------------------------------------
    expect_reject("no header row", "0,70\n1,60\n", tmp)
    expect_reject("only one point",
                  "concentration,ift\n0,70.55\n", tmp)
    expect_reject("non-numeric data",
                  "concentration,ift\n0,70.55\nabc,50\n", tmp)
    expect_reject("NaN data",
                  "concentration,ift\n0,70.55\nnan,50\n", tmp)
    expect_reject("duplicate concentration",
                  "concentration,ift\n0,70\n0,60\n5,50\n", tmp)
    expect_reject("formula string not executed",
                  "concentration,ift\n0,70\n=cmd|calc,50\n", tmp)
    expect_reject("negative override",
                  GOOD.replace("c_sat,", "c_sat,-5"), tmp)

    print(f"\n{_passed} passed, {_failed} failed")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()

"""
Tests for core.presets (settings presets + session state).

Runs against a throwaway user folder (IFT_PRESETS_HOME), never the real
%APPDATA% one.  Run from the repo root:
    conda run -n ift python tests/test_presets.py
"""

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TMP = tempfile.mkdtemp(prefix="ift_presets_test_")
os.environ["IFT_PRESETS_HOME"] = TMP
from core import presets as P  # noqa: E402  (env must be set first)

failures = []


def check(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        failures.append(msg)


def raises(fn, *a, contains="", **kw):
    try:
        fn(*a, **kw)
    except P.PresetError as e:
        return contains.lower() in str(e).lower()
    return False


NEEDLE = dict(od_mm=1.58, id_mm=0.40, od_tol_pct=1.0, notes="test")
FLUID = dict(rho_in=1045.0, rho_out=160.81, rho_in_tol_pct=0.1,
             rho_out_tol_pct=2.0, T_C=27.0, P_MPa=5.67, notes="")

try:
    print("Built-in presets")
    for kind, n_min in (("needle", 3), ("fluid", 7)):
        items, problems = P.list_presets(kind)
        check(len([p for p in items if p.builtin]) >= n_min and not problems,
              f"{kind}: {len(items)} built-ins load cleanly ({problems or 'no problems'})")
    yang = P.find_preset("fluid", "co₂ / weyburn brine, 5.67 mpa, 27 °c")
    check(yang is not None and abs(yang.values["rho_in"] - yang.values["rho_out"] - 884.19) < 0.01,
          "Yang 5.67 MPa preset found case-insensitively, Δρ = 884.19 kg/m³")

    print("Validation")
    check(raises(P.validate_values, "needle", dict(NEEDLE, id_mm=2.0), contains="smaller"),
          "ID ≥ OD rejected")
    check(raises(P.validate_values, "needle", dict(NEEDLE, od_mm=-1), contains="> 0"),
          "negative OD rejected")
    check(raises(P.validate_values, "fluid", dict(FLUID, rho_out=float("nan")), contains="finite"),
          "NaN density rejected")
    check(raises(P.validate_values, "fluid", dict(FLUID, rho_out=1045.0), contains="Δρ = 0"),
          "equal densities rejected (Δρ = 0)")
    check(raises(P.validate_values, "fluid", dict(FLUID, rho_in="=1+1"), contains="number"),
          "formula-like string rejected, never evaluated")
    check(raises(P.validate_values, "fluid", dict(FLUID, rho_in=True), contains="number"),
          "boolean rejected as a number")
    check(raises(P.validate_values, "needle", {k: v for k, v in NEEDLE.items() if k != "od_mm"},
                 contains="missing"), "missing required field rejected")
    check(P.validate_values("needle", dict(NEEDLE, id_mm=None))["id_mm"] is None,
          "optional ID may be empty")
    check(raises(P.validate_name, "   ", contains="empty"), "blank name rejected")
    check(raises(P.validate_name, "a\x00b", contains="control"), "control characters rejected")

    print("Save / load / overwrite / delete")
    p = P.save_user_preset("needle", "Lab needle A", NEEDLE)
    check(p.path.parent == Path(TMP) / "presets" / "needle" and p.path.is_file(),
          f"saved into the user folder ({p.path.name})")
    check(P.find_preset("needle", "lab needle a").values == P.validate_values("needle", NEEDLE),
          "round-trip values identical")
    check(raises(P.save_user_preset, "needle", "Lab needle A", NEEDLE, contains="already exists"),
          "duplicate name refused without overwrite")
    P.save_user_preset("needle", "Lab needle A", dict(NEEDLE, od_mm=1.6), overwrite=True)
    check(P.find_preset("needle", "Lab needle A").values["od_mm"] == 1.6, "overwrite updates value")
    check(raises(P.save_user_preset, "fluid", "Water / air, 20 °C", FLUID, contains="built-in"),
          "cannot shadow a built-in name")
    check(raises(P.delete_user_preset, "fluid", "Water / air, 20 °C", contains="built in"),
          "cannot delete a built-in")
    evil = P.save_user_preset("fluid", "../../escape attempt", FLUID)
    check(evil.path.parent == Path(TMP) / "presets" / "fluid",
          f"path-traversal name stays inside the folder ({evil.path.name})")
    P.delete_user_preset("fluid", "../../escape attempt")
    check(P.find_preset("fluid", "../../escape attempt") is None, "user preset deleted")
    leftovers = list((Path(TMP) / "presets" / "needle").glob(".tmp_*"))
    check(not leftovers, "no temp files left behind by atomic writes")

    print("Corrupt files are skipped, not fatal")
    bad = Path(TMP) / "presets" / "fluid" / "broken.json"
    bad.write_text("{not json", encoding="utf-8")
    (Path(TMP) / "presets" / "fluid" / "wrongkind.json").write_text(json.dumps(
        dict(schema=1, kind="needle", name="Sneaky", values=NEEDLE)), encoding="utf-8")
    (Path(TMP) / "presets" / "fluid" / "big.json").write_text(" " * (P.MAX_FILE_BYTES + 1))
    items, problems = P.list_presets("fluid")
    check(len(problems) == 3 and all(not p.name == "Sneaky" for p in items),
          f"3 bad files reported and skipped; {len(items)} good fluid presets still listed")

    print("Import / export")
    exp = Path(TMP) / "exported.json"
    P.export_preset(P.find_preset("needle", "Lab needle A"), exp)
    P.delete_user_preset("needle", "Lab needle A")
    imp = P.import_preset(exp)
    check(imp.name == "Lab needle A" and imp.values["od_mm"] == 1.6, "export → import round-trip")
    check(raises(P.import_preset, exp, contains="already exists"), "re-import asks before replacing")
    check(raises(P.import_preset, bad, contains="not valid json"), "importing a broken file fails cleanly")

    print("Session state")
    st = P.load_state()
    check(st["fluid"]["rho_in"] == 998.2 and st["needle"]["od_mm"] == 0.7176 and st["ts_interval"] == 1.0,
          "no state file → factory water/air + 22G needle")
    P.save_state("Lab needle A", dict(NEEDLE, od_mm=1.6), None, dict(FLUID, rho_out=150.0), 2.5)
    st = P.load_state()
    check(st["needle_name"] == "Lab needle A" and st["fluid_name"] is None
          and st["fluid"]["rho_out"] == 150.0 and st["ts_interval"] == 2.5,
          "state round-trip incl. custom (unsaved) fluid values")
    (Path(TMP) / "state.json").write_text("garbage", encoding="utf-8")
    check(P.load_state()["fluid"]["rho_in"] == 998.2, "corrupt state file → factory defaults")
    check(P.values_equal("fluid", FLUID, dict(FLUID)) and
          not P.values_equal("fluid", FLUID, dict(FLUID, rho_out=160.82)),
          "values_equal detects a changed density")
finally:
    shutil.rmtree(TMP, ignore_errors=True)

print()
if failures:
    print(f"{len(failures)} check(s) FAILED")
    sys.exit(1)
print("All checks passed.")

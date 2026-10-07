"""
core.presets
============

Named, reusable settings so physical parameters survive between sessions.

Two independent kinds, because they change at different times:

    needle  – hardware: outer/inner diameter, OD tolerance
    fluid   – the experiment's fluid system: inner/outer density with
              tolerances, plus descriptive temperature / pressure / notes

e.g. one needle used at six pressures = 1 needle preset + 6 fluid presets.

Storage
-------
    built-in  <repo>/presets/{needle,fluid}/*.json      read-only, shipped
    user      %APPDATA%/IFTProperties/presets/{needle,fluid}/*.json
    state     %APPDATA%/IFTProperties/state.json        last selection + values

``IFT_PRESETS_HOME`` overrides the user root (tests).  Every file is fully
validated on load (finite, positive, physically consistent numbers; size
and type checks) and written atomically (temp file + os.replace), so a
crash mid-save never leaves a half-written preset.  A corrupt file is
skipped and reported, never fatal.
"""

import json
import math
import os
import re
import tempfile
from pathlib import Path

SCHEMA_VERSION = 1
KINDS = ('needle', 'fluid')
MAX_FILE_BYTES = 64 * 1024
MAX_NAME_LEN = 80
MAX_NOTES_LEN = 2000

BUILTIN_DIR = Path(__file__).resolve().parent.parent / 'presets'

# Field specs: name -> (required, lower bound (exclusive unless noted), upper)
_NUM = {
    'needle': {
        'od_mm':      (True,  0.0, 50.0),
        'id_mm':      (False, 0.0, 50.0),
        'od_tol_pct': (True,  None, 100.0),     # >= 0
    },
    'fluid': {
        'rho_in':          (True,  0.0, 30000.0),
        'rho_out':         (True,  0.0, 30000.0),
        'rho_in_tol_pct':  (True,  None, 100.0),
        'rho_out_tol_pct': (True,  None, 100.0),
        'T_C':             (False, -273.15, 2000.0),
        'P_MPa':           (False, None, 1000.0),
    },
}

# Values shipped in the app before presets existed (water/air, 22G needle).
FACTORY = {
    'needle': dict(od_mm=0.7176, id_mm=None, od_tol_pct=1.0, notes=''),
    'fluid':  dict(rho_in=998.2, rho_out=1.204, rho_in_tol_pct=0.05,
                   rho_out_tol_pct=2.0, T_C=20.0, P_MPa=0.101325, notes=''),
}
# Built-in presets matching FACTORY's numbers (selected on a first run).
FACTORY_NAMES = {'needle': "22G needle (OD 0.7176 mm)",
                 'fluid': "Water / air, 20 °C"}


class PresetError(ValueError):
    """A preset file or value set is invalid; message is user-facing."""


# ---------------------------------------------------------------------------
# Locations
# ---------------------------------------------------------------------------
def user_root():
    env = os.environ.get('IFT_PRESETS_HOME')
    if env:
        return Path(env)
    base = os.environ.get('APPDATA') or str(Path.home() / '.config')
    return Path(base) / 'IFTProperties'


def _user_dir(kind):
    return user_root() / 'presets' / kind


def _state_path():
    return user_root() / 'state.json'


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _check_kind(kind):
    if kind not in KINDS:
        raise PresetError(f"Unknown preset kind {kind!r} (expected needle or fluid).")


def validate_name(name):
    if not isinstance(name, str):
        raise PresetError("Preset name must be text.")
    name = name.strip()
    if not name:
        raise PresetError("Preset name is empty.")
    if len(name) > MAX_NAME_LEN:
        raise PresetError(f"Preset name is longer than {MAX_NAME_LEN} characters.")
    if any(ord(c) < 32 for c in name):
        raise PresetError("Preset name contains control characters.")
    return name


def validate_values(kind, values):
    """Return a clean copy of ``values`` for ``kind`` or raise PresetError."""
    _check_kind(kind)
    if not isinstance(values, dict):
        raise PresetError("Preset values must be a JSON object.")
    out = {}
    for key, (required, lo, hi) in _NUM[kind].items():
        v = values.get(key)
        if v is None or v == '':
            if required:
                raise PresetError(f"Missing required value '{key}'.")
            out[key] = None
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise PresetError(f"'{key}' must be a number, got {v!r}.")
        v = float(v)
        if not math.isfinite(v):
            raise PresetError(f"'{key}' must be finite, got {v}.")
        if lo is None:
            if v < 0:
                raise PresetError(f"'{key}' must be ≥ 0, got {v:g}.")
        elif v <= lo:
            raise PresetError(f"'{key}' must be > {lo:g}, got {v:g}.")
        if v > hi:
            raise PresetError(f"'{key}' must be ≤ {hi:g}, got {v:g}.")
        out[key] = v

    if kind == 'needle':
        if out['id_mm'] is not None and out['id_mm'] >= out['od_mm']:
            raise PresetError(
                f"Needle ID ({out['id_mm']:g} mm) must be smaller than OD "
                f"({out['od_mm']:g} mm).")
    elif out['rho_in'] == out['rho_out']:
        raise PresetError("Inner and outer densities are equal, so Δρ = 0 "
                          "and no interfacial tension can be computed.")
    notes = values.get('notes') or ''
    if not isinstance(notes, str):
        raise PresetError("'notes' must be text.")
    if len(notes) > MAX_NOTES_LEN:
        raise PresetError(f"'notes' is longer than {MAX_NOTES_LEN} characters.")
    out['notes'] = notes
    return out


def _parse_file(path):
    """Read + validate one preset file → (kind, name, values)."""
    path = Path(path)
    try:
        size = path.stat().st_size
    except OSError as e:
        raise PresetError(f"Cannot read {path.name}: {e}")
    if size > MAX_FILE_BYTES:
        raise PresetError(f"{path.name} is too large to be a preset "
                          f"({size} bytes > {MAX_FILE_BYTES}).")
    try:
        data = json.loads(path.read_text(encoding='utf-8-sig'))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise PresetError(f"{path.name} is not valid JSON: {e}")
    if not isinstance(data, dict):
        raise PresetError(f"{path.name}: top level must be a JSON object.")
    if data.get('schema') != SCHEMA_VERSION:
        raise PresetError(f"{path.name}: unsupported schema {data.get('schema')!r} "
                          f"(expected {SCHEMA_VERSION}).")
    kind = data.get('kind')
    _check_kind(kind)
    name = validate_name(data.get('name'))
    values = validate_values(kind, data.get('values'))
    return kind, name, values


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------
class Preset:
    __slots__ = ('kind', 'name', 'values', 'builtin', 'path')

    def __init__(self, kind, name, values, builtin, path):
        self.kind, self.name, self.values = kind, name, values
        self.builtin, self.path = builtin, path

    def __repr__(self):
        return f"Preset({self.kind}, {self.name!r}, builtin={self.builtin})"


def list_presets(kind):
    """All valid presets of ``kind``: built-ins first, then the user's.

    Returns (presets, problems) — problems are human-readable messages for
    files that were skipped.  A user preset may not shadow a built-in name.
    """
    _check_kind(kind)
    presets, problems, seen = [], [], set()
    for builtin, folder in ((True, BUILTIN_DIR / kind), (False, _user_dir(kind))):
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob('*.json')):
            try:
                k, name, values = _parse_file(path)
            except PresetError as e:
                problems.append(str(e))
                continue
            if k != kind:
                problems.append(f"{path.name}: is a {k} preset in the {kind} folder.")
                continue
            if name.casefold() in seen:
                problems.append(f"{path.name}: duplicate preset name '{name}' skipped.")
                continue
            seen.add(name.casefold())
            presets.append(Preset(kind, name, values, builtin, path))
    return presets, problems


def find_preset(kind, name):
    for p in list_presets(kind)[0]:
        if p.name.casefold() == (name or '').casefold():
            return p
    return None


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------
def _slug(name):
    """Filesystem-safe file stem: lowercase ASCII, digits, '.', '-'."""
    s = re.sub(r'[^a-z0-9.]+', '-', name.lower()).strip('-.')
    return (s or 'preset')[:60]


def _atomic_write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix='.tmp_', suffix='.json')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def save_user_preset(kind, name, values, overwrite=False):
    """Validate and save a user preset; returns the saved Preset.

    Refuses to shadow a built-in.  An existing user preset of the same name
    is replaced only with ``overwrite=True``.
    """
    _check_kind(kind)
    name = validate_name(name)
    clean = validate_values(kind, values)
    existing = find_preset(kind, name)
    if existing is not None:
        if existing.builtin:
            raise PresetError(f"'{name}' is a built-in preset; choose another name.")
        if not overwrite:
            raise PresetError(f"A preset named '{name}' already exists.")
        path = existing.path
    else:
        folder = _user_dir(kind)
        path = folder / f"{_slug(name)}.json"
        n = 2
        while path.exists():
            path = folder / f"{_slug(name)}_{n}.json"
            n += 1
    _atomic_write_json(path, dict(schema=SCHEMA_VERSION, kind=kind,
                                  name=name, values=clean))
    return Preset(kind, name, clean, False, path)


def delete_user_preset(kind, name):
    p = find_preset(kind, name)
    if p is None:
        raise PresetError(f"No preset named '{name}'.")
    if p.builtin:
        raise PresetError(f"'{name}' is built in and cannot be deleted.")
    p.path.unlink()


def import_preset(path, overwrite=False):
    """Copy an external preset file into the user's presets (validated)."""
    kind, name, values = _parse_file(path)
    return save_user_preset(kind, name, values, overwrite=overwrite)


def export_preset(preset, dest):
    _atomic_write_json(Path(dest), dict(schema=SCHEMA_VERSION, kind=preset.kind,
                                        name=preset.name, values=preset.values))


# ---------------------------------------------------------------------------
# Session state (last selection + the values actually in use)
# ---------------------------------------------------------------------------
def load_state():
    """Last session's selection and values, or factory defaults.

    Returns dict(needle_name, fluid_name, needle=values, fluid=values,
    ts_interval).  Anything missing or invalid falls back to FACTORY.
    """
    st = dict(needle_name=FACTORY_NAMES['needle'], fluid_name=FACTORY_NAMES['fluid'],
              needle=dict(FACTORY['needle']), fluid=dict(FACTORY['fluid']),
              ts_interval=1.0)
    for kind in KINDS:
        # Take the shipped preset's values (incl. notes) so a first run does
        # not show the selected preset as "Modified".
        p = find_preset(kind, FACTORY_NAMES[kind])
        if p is not None:
            st[kind] = dict(p.values)
    path = _state_path()
    try:
        if not path.is_file() or path.stat().st_size > MAX_FILE_BYTES:
            return st
        data = json.loads(path.read_text(encoding='utf-8-sig'))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return st
    if not isinstance(data, dict):
        return st
    for kind in KINDS:
        try:
            st[kind] = validate_values(kind, data.get(kind))
        except PresetError:
            pass
        name = data.get(f'{kind}_name')
        st[f'{kind}_name'] = name if isinstance(name, str) and name.strip() else None
    ts = data.get('ts_interval')
    if isinstance(ts, (int, float)) and not isinstance(ts, bool) \
            and math.isfinite(ts) and ts > 0:
        st['ts_interval'] = float(ts)
    return st


def save_state(needle_name, needle_values, fluid_name, fluid_values, ts_interval):
    """Remember the current selection and values for the next session."""
    _atomic_write_json(_state_path(), dict(
        schema=SCHEMA_VERSION,
        needle_name=needle_name, needle=validate_values('needle', needle_values),
        fluid_name=fluid_name, fluid=validate_values('fluid', fluid_values),
        ts_interval=float(ts_interval)))


def values_equal(kind, a, b, rel=1e-9):
    """True when two value dicts of ``kind`` describe the same preset."""
    for key in _NUM[kind]:
        x, y = a.get(key), b.get(key)
        if (x is None) != (y is None):
            return False
        if x is not None and abs(x - y) > rel * max(1.0, abs(x), abs(y)):
            return False
    return (a.get('notes') or '') == (b.get('notes') or '')

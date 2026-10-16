"""
core.session
============

A session folder holds everything one experiment produced, in a form that
can be opened again without the app remembering how it was made::

    <name>/
      session.json      how the frames were made and measured (this module)
      log.csv           one row per logged frame (core.frame_log)
      frames/           the logged frames: cropped, grayscale, lossless PNG,
                        named <frame index>_t<seconds>.png
      calibration.csv   copy of the gamma(c) curve, if one is used (Yang)
      ift_curve.csv     copy of an IFT-vs-time curve from a file, if one is
                        used for Yang instead of the logged IFT (for example
                        when only one image of the drop was logged)
      volume_curve.csv  copy of a volume-vs-time curve from a file, if one is
                        used for Bagalkot instead of the logged volume
      analyses/         settings and results of each analysis (JSON)

Logging writes the folder (:meth:`SessionFolder.create`, then
:meth:`~SessionFolder.log_frame` / :meth:`~SessionFolder.log_video`, then
:meth:`~SessionFolder.finish`).  Opening it (:meth:`SessionFolder.open`)
gives back everything the analyses need.

Source of truth
---------------
``frames/`` and ``session.json`` are the source of truth; ``log.csv`` is
derived from them and can always be rebuilt (:meth:`~SessionFolder.rebuild_log`),
for example with other columns, or after the file was edited outside the
app.  ``session.json`` stores a SHA-256 of ``log.csv`` so such an edit is
noticed on open.  While logging runs that hash is empty, so a session cut
short by a crash is recognised too; the rows written up to then are kept,
because each row is appended to the file as soon as its frame is fitted.

Frames and coordinates
----------------------
Frames are stored after the crop steps but before the pixel steps
(threshold), so the threshold can be changed later and the frames
re-logged.  When the user thresholded before cropping, the stored frame
differs from what was on screen only within the blur kernel of the crop
edge, far from the drop.  ``baseline.y_px`` is in the coordinates of the
stored (cropped) frame, as the user picks it on the cropped image.

Nothing here touches Qt.
"""

import datetime
import hashlib
import json
import math
import os
import re
import tempfile
from collections import namedtuple
from dataclasses import dataclass, field, fields
from pathlib import Path

import cv2
import numpy as np

from core import frame_log as fl
from core.frame_series import (FrameSeriesCancelled, analyse_frame, apply_prep,
                               frames_for_times, read_frames, video_info)

SCHEMA_VERSION = 1
SESSION_FILE = 'session.json'
LOG_FILE = 'log.csv'
FRAMES_DIR = 'frames'
CALIBRATION_FILE = 'calibration.csv'
IFT_CURVE_FILE = 'ift_curve.csv'
VOLUME_CURVE_FILE = 'volume_curve.csv'
ANALYSES_DIR = 'analyses'

SOURCE_KINDS = ('video', 'camera', 'image')
MIN_SERIES_FRAMES = 5           # same minimum as core.swelling
YANG_MIN_POINTS = 10            # QC-passed IFT points the Yang fit needs
MAX_SESSION_BYTES = 1_000_000
MAX_LOG_BYTES = 100_000_000

_FRAME_NAME = re.compile(r'^(\d{6,})_t(\d+(?:\.\d+)?)\.png$')
_ANALYSIS_NAME = re.compile(r'^[a-z][a-z0-9_]{0,40}$')

Problem = namedtuple('Problem', 'code message fixed_by_rebuild')


class SessionError(ValueError):
    """A session that can't be created, opened or used as asked."""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _now():
    return datetime.datetime.now().astimezone().isoformat(timespec='seconds')


def _atomic_write_bytes(path, data):
    """Write via a temp file in the same folder + os.replace, so a crash
    never leaves a half-written file."""
    path = Path(path)
    fd, tmp = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _jsonable(obj):
    """Plain JSON types: numpy scalars/arrays unwrapped, NaN/Inf -> None."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, Path):
        return str(obj)
    if obj is None or isinstance(obj, str):
        return obj
    raise SessionError(f"cannot store a {type(obj).__name__} in JSON")


def _number(d, key, where, lo=None, hi=None, required=True, default=None):
    v = d.get(key, default)
    if v is None:
        if required:
            raise SessionError(f"{where}: '{key}' is missing")
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise SessionError(f"{where}: '{key}' must be a number, got {v!r}")
    v = float(v)
    if not math.isfinite(v):
        raise SessionError(f"{where}: '{key}' must be finite")
    if lo is not None and v <= lo:
        raise SessionError(f"{where}: '{key}' must be > {lo:g}, got {v:g}")
    if hi is not None and v > hi:
        raise SessionError(f"{where}: '{key}' must be <= {hi:g}, got {v:g}")
    return v


def split_prep(prep_ops):
    """(crop steps, pixel steps): crops are applied before a frame is
    stored, pixel steps (threshold) every time it is read."""
    crops = [op for op in prep_ops if op[0] == 'crop']
    pixel = [op for op in prep_ops if op[0] != 'crop']
    return crops, pixel


def _normalise_prep(prep_ops):
    out = []
    for item in prep_ops or ():
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise SessionError(f"preparation step {item!r} is not (name, value)")
        op, arg = item
        if op == 'crop':
            if not isinstance(arg, (list, tuple)) or len(arg) != 4:
                raise SessionError(f"crop needs (x, y, w, h), got {arg!r}")
            x, y, w, h = (int(v) for v in arg)
            if w <= 0 or h <= 0 or x < 0 or y < 0:
                raise SessionError(f"crop ({x}, {y}, {w}, {h}) is empty or negative")
            out.append(('crop', (x, y, w, h)))
        elif op == 'threshold':
            if isinstance(arg, bool) or not isinstance(arg, (int, float)) or not 0 <= arg <= 255:
                raise SessionError(f"threshold must be 0-255, got {arg!r}")
            out.append(('threshold', int(arg)))
        else:
            raise SessionError(f"unknown preparation step {op!r}")
    return out


def frame_file_name(frame_index, t_s):
    return f"{int(frame_index):06d}_t{t_s:.3f}.png"


def _to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def _write_png(path, img):
    ok, buf = cv2.imencode('.png', img)
    if not ok:
        raise SessionError(f"could not encode {Path(path).name} as PNG")
    _atomic_write_bytes(path, buf.tobytes())


def _read_png(path):
    # imdecode on bytes: cv2.imread can't open non-ASCII paths on Windows.
    img = cv2.imdecode(np.frombuffer(Path(path).read_bytes(), np.uint8),
                       cv2.IMREAD_UNCHANGED)
    if img is None:
        raise SessionError(f"{Path(path).name} is not a readable image")
    return img


# ---------------------------------------------------------------------------
# session.json
# ---------------------------------------------------------------------------
@dataclass
class Session:
    """What session.json holds.

    source    kind ('video' | 'camera' | 'image'), name, path, fps,
              n_frames, width, height, t_start_s (video: when the drop
              formed, frames before it are not logged)
    scale     pixel_to_m, method ('auto' | 'manual' | 'figure'), reference
    baseline  y_px (stored-frame coordinates), method
    needle    od_mm, tol_rel, name, id_mm (optional; the diffusion model's
              domain, default 70 % of od_mm)
    fluids    density_inner, density_outer (kg/m3), rho_in_tol_rel,
              rho_out_tol_rel, name, pressure_MPa, temperature_C
    prep_ops  [('crop', (x, y, w, h)) | ('threshold', value), ...] in the
              order the user made them
    sampling  how the times were chosen (core.frame_series.sample_times)
    columns   full log header (core.frame_log.column_keys)
    analyses  enabled analyses; their columns are locked into the log
    """
    source: dict
    scale: dict
    baseline: dict
    needle: dict
    fluids: dict
    prep_ops: list = field(default_factory=list)
    sampling: dict = field(default_factory=dict)
    columns: list = None
    analyses: list = field(default_factory=list)
    calibration: str = None
    notes: str = ''
    created: str = ''
    log_sha256: str = None
    n_logged: int = 0
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self):
        self.prep_ops = _normalise_prep(self.prep_ops)
        self.analyses = list(self.analyses or ())
        if self.columns is None:
            self.columns = fl.column_keys(analyses=self.analyses,
                                          camera=self.source.get('kind') == 'camera')
        self.columns = list(self.columns)
        self.validate()

    def validate(self):
        if not isinstance(self.schema_version, int) or self.schema_version < 1:
            raise SessionError(f"bad schema_version {self.schema_version!r}")
        if self.schema_version > SCHEMA_VERSION:
            raise SessionError(
                f"this session was saved by a newer version of the app "
                f"(format {self.schema_version}, this app reads up to "
                f"{SCHEMA_VERSION}); update the app to open it")
        if self.source.get('kind') not in SOURCE_KINDS:
            raise SessionError(f"source kind must be one of {SOURCE_KINDS}, "
                               f"got {self.source.get('kind')!r}")
        _number(self.scale, 'pixel_to_m', 'scale', lo=1e-9, hi=1e-2)
        _number(self.baseline, 'y_px', 'baseline', lo=-1)
        _number(self.needle, 'od_mm', 'needle', lo=0, hi=100)
        _number(self.needle, 'tol_rel', 'needle', lo=-1e-12, hi=1, required=False)
        id_mm = _number(self.needle, 'id_mm', 'needle', lo=0, hi=100, required=False)
        if id_mm is not None and id_mm >= self.needle['od_mm']:
            raise SessionError("needle: the inner diameter must be below the outer diameter")
        rin = _number(self.fluids, 'density_inner', 'fluids', lo=0, hi=30000)
        rout = _number(self.fluids, 'density_outer', 'fluids', lo=0, hi=30000)
        if rin == rout:
            raise SessionError("fluids: inner and outer densities are equal, "
                               "so no interfacial tension can be computed")
        for k in ('rho_in_tol_rel', 'rho_out_tol_rel'):
            _number(self.fluids, k, 'fluids', lo=-1e-12, hi=1, required=False)
        try:
            fl.FrameLog(self.columns)
            need = fl.required_columns(self.analyses)
        except fl.FrameLogError as e:
            raise SessionError(str(e)) from None
        missing = [k for k in need if k not in self.columns]
        if missing:
            raise SessionError(f"the enabled analyses need column(s) "
                               f"{', '.join(missing)} in the log")
        if self.calibration not in (None, CALIBRATION_FILE):
            raise SessionError(f"calibration must be '{CALIBRATION_FILE}' or empty")

    def analysis_params(self):
        """The ``params`` dict of :func:`core.frame_series.analyse_frame`."""
        f, n = self.fluids, self.needle
        return dict(baseline_y=float(self.baseline['y_px']),
                    pixel_to_m=float(self.scale['pixel_to_m']),
                    density_inner=float(f['density_inner']),
                    density_outer=float(f['density_outer']),
                    needle_tol_rel=float(n.get('tol_rel', 0.01)),
                    rho_in_tol_rel=float(f.get('rho_in_tol_rel', 0.0005)),
                    rho_out_tol_rel=float(f.get('rho_out_tol_rel', 0.02)),
                    needle_diameter_m=float(n['od_mm']) / 1000.0)

    def to_dict(self):
        return _jsonable({f.name: getattr(self, f.name) for f in fields(self)})

    @classmethod
    def from_dict(cls, d):
        if not isinstance(d, dict):
            raise SessionError("session.json does not hold an object")
        known = {f.name for f in fields(cls)}
        kw = {k: v for k, v in d.items() if k in known}
        for k in ('source', 'scale', 'baseline', 'needle', 'fluids'):
            if not isinstance(kw.get(k), dict):
                raise SessionError(f"session.json: '{k}' is missing or not an object")
        try:
            return cls(**kw)
        except TypeError as e:
            raise SessionError(f"session.json: {e}") from None


# ---------------------------------------------------------------------------
# The folder
# ---------------------------------------------------------------------------
class SessionFolder:
    """A session folder on disk plus its log in memory.

    ``problems`` lists what :meth:`open` found wrong (missing frames, an
    edited or unfinished log...), each with whether :meth:`rebuild_log`
    fixes it.
    """

    def __init__(self, path, session, log, problems=()):
        self.path = Path(path)
        self.session = session
        self.log = log
        self.problems = list(problems)
        self._frames = self._scan_frames()

    # -- paths ---------------------------------------------------------------
    @property
    def session_path(self):
        return self.path / SESSION_FILE

    @property
    def log_path(self):
        return self.path / LOG_FILE

    @property
    def frames_dir(self):
        return self.path / FRAMES_DIR

    @property
    def calibration_path(self):
        return self.path / CALIBRATION_FILE if self.session.calibration else None

    @property
    def ift_curve_path(self):
        """The IFT-vs-time curve copied in from a file, or None."""
        p = self.path / IFT_CURVE_FILE
        return p if p.is_file() else None

    @property
    def volume_curve_path(self):
        """The volume-vs-time curve copied in from a file, or None."""
        p = self.path / VOLUME_CURVE_FILE
        return p if p.is_file() else None

    def _scan_frames(self):
        out = {}
        if self.frames_dir.is_dir():
            for p in self.frames_dir.iterdir():
                m = _FRAME_NAME.match(p.name)
                if m:
                    out[int(m.group(1))] = p
        return out

    # -- create and log --------------------------------------------------------
    @classmethod
    def create(cls, path, session):
        """Start a new session folder.  ``path`` must not exist or be empty."""
        path = Path(path)
        if path.exists() and (not path.is_dir() or any(path.iterdir())):
            raise SessionError(f"{path} already exists and is not an empty folder")
        (path / FRAMES_DIR).mkdir(parents=True, exist_ok=True)
        (path / ANALYSES_DIR).mkdir(exist_ok=True)
        session.created = session.created or _now()
        session.log_sha256 = None
        session.n_logged = 0
        folder = cls(path, session, fl.FrameLog(session.columns))
        folder._write_session()
        _atomic_write_bytes(folder.log_path, fl.format_header(
            session.columns, folder._header_lines()).encode('utf-8'))
        return folder

    def _header_lines(self):
        s = self.session
        f = s.fluids
        lines = [f"Drop Analysis frame log, session created {s.created}",
                 f"source: {s.source.get('kind')} {s.source.get('name', '')}".rstrip(),
                 f"scale: {s.scale['pixel_to_m'] * 1e6:.4g} um/px "
                 f"({s.scale.get('method', 'unknown')}); baseline y = {s.baseline['y_px']:g} px",
                 f"fluids: {f.get('name', '')} inner {f['density_inner']:g} kg/m3, "
                 f"outer {f['density_outer']:g} kg/m3"]
        if f.get('pressure_MPa') is not None or f.get('temperature_C') is not None:
            lines.append(f"conditions: {f.get('pressure_MPa', '?')} MPa, "
                         f"{f.get('temperature_C', '?')} C")
        lines.append("this file is rebuilt from frames/ and session.json; "
                     "edits made here are not kept")
        return lines

    def _write_session(self):
        text = json.dumps(self.session.to_dict(), indent=2, ensure_ascii=False)
        _atomic_write_bytes(self.session_path, text.encode('utf-8'))

    def log_frame(self, raw_frame, t_s, frame_index, wall_clock=None):
        """Store one frame and append its measurement to the log.

        ``raw_frame`` is the frame as it comes from the source (BGR or
        grayscale); the session's crop is applied before storing and its
        threshold before fitting.  Returns the analyse_frame record."""
        t_s = float(t_s)
        if not math.isfinite(t_s) or t_s < 0:
            raise SessionError(f"frame time must be >= 0 s, got {t_s}")
        frame_index = int(frame_index)
        if frame_index in self._frames:
            raise SessionError(f"frame {frame_index} is already logged")
        crops, pixel = split_prep(self.session.prep_ops)
        stored = _to_gray(apply_prep(raw_frame, crops))
        p = self.frames_dir / frame_file_name(frame_index, t_s)
        _write_png(p, stored)
        self._frames[frame_index] = p
        rec = self._measure(stored, pixel)
        rec.update(t_s=t_s, frame_index=frame_index, wall_clock=wall_clock or '')
        row = self.log.append_record(rec)
        with open(self.log_path, 'a', newline='', encoding='utf-8') as f:
            f.write(fl.format_row(self.log.columns, row))
        self.session.n_logged = len(self.log)
        return rec

    def _measure(self, stored_gray, pixel_ops):
        img = apply_prep(cv2.cvtColor(stored_gray, cv2.COLOR_GRAY2BGR), pixel_ops)
        # Each frame starts from its own guesses (apex circle, DS/DE): with
        # the ADSA-P fit that is faster than the previous frame's solution.
        return analyse_frame(img, self.session.analysis_params())

    def measure_frame(self, frame_index):
        """Fit one stored frame again, exactly as the logger did; nothing is
        written.  Returns the analyse_frame record."""
        p = self._frames.get(int(frame_index))
        if p is None:
            raise SessionError(f"frame {frame_index} is not stored in this session")
        return self._measure(_to_gray(_read_png(p)), split_prep(self.session.prep_ops)[1])

    def measure_image(self, raw_frame):
        """Fit an image taken with this session's set-up (same camera,
        needle and magnification): the session's crop, threshold, scale,
        baseline and fluids are applied; nothing is stored or logged.
        Returns the analyse_frame record with the cropped frame under
        'stored_frame'."""
        src = self.session.source
        h, w = raw_frame.shape[:2]
        if src.get('width') and src.get('height') and (w, h) != (int(src['width']), int(src['height'])):
            raise SessionError(f"the image is {w} x {h} px but this session's frames are "
                               f"{src['width']} x {src['height']} px; its scale and baseline "
                               f"would not apply")
        crops, pixel = split_prep(self.session.prep_ops)
        stored = _to_gray(apply_prep(raw_frame, crops))
        rec = self._measure(stored, pixel)
        rec['stored_frame'] = stored
        return rec

    def log_image(self, img):
        """A single image is a session with one frame at t = 0."""
        return self.log_frame(img, 0.0, 0)

    def log_video(self, sample_t, progress_cb=None, is_cancelled=None):
        """Log the video frames nearest to ``sample_t`` (s after the
        source's ``t_start_s``).  ``progress_cb(i, n, rec)`` after each
        frame; ``is_cancelled()`` polled between frames (raises
        FrameSeriesCancelled; frames logged so far stay in the folder).
        Returns the number of frames logged."""
        src = self.session.source
        if src.get('kind') != 'video':
            raise SessionError("log_video needs a video source")
        info = video_info(src['path'])
        idx, t_act = frames_for_times(sample_t, info['fps'],
                                      float(src.get('t_start_s', 0.0)), info['n_frames'])
        keep = t_act >= -0.5 / info['fps']
        idx, t_act = idx[keep], np.maximum(t_act[keep], 0.0)
        t_of = dict(zip(idx.tolist(), t_act.tolist()))
        n, done = len(idx), 0
        for i, frame in read_frames(src['path'], idx):
            if is_cancelled is not None and is_cancelled():
                raise FrameSeriesCancelled()
            rec = self.log_frame(frame, t_of[i], i)
            done += 1
            if progress_cb is not None:
                progress_cb(done, n, rec)
        return done

    def finish(self):
        """Seal the log: record its hash so later edits are noticed."""
        self.session.log_sha256 = sha256_file(self.log_path)
        self.session.n_logged = len(self.log)
        self._write_session()
        self.problems = [p for p in self.problems if p.code != 'log_unfinished']

    # -- open -------------------------------------------------------------------
    @classmethod
    def open(cls, path):
        """Open a session folder and check it.  Raises SessionError when
        session.json is missing or invalid; anything else is reported in
        ``problems``."""
        path = Path(path)
        sp = path / SESSION_FILE
        if not sp.is_file():
            raise SessionError(f"{path} is not a session folder (no {SESSION_FILE})")
        if sp.stat().st_size > MAX_SESSION_BYTES:
            raise SessionError(f"{SESSION_FILE} is too large to be a session file")
        try:
            d = json.loads(sp.read_text(encoding='utf-8-sig'))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise SessionError(f"{SESSION_FILE} is not valid JSON: {e}") from None
        session = Session.from_dict(d)

        problems = []
        lp = path / LOG_FILE
        log = None
        if not lp.is_file():
            problems.append(Problem('log_missing', "log.csv is missing; rebuild it "
                                    "from the stored frames.", True))
        elif lp.stat().st_size > MAX_LOG_BYTES:
            problems.append(Problem('log_unreadable', "log.csv is too large to be "
                                    "a frame log; rebuild it from the frames.", True))
        else:
            try:
                log = fl.FrameLog.from_text(lp.read_text(encoding='utf-8-sig'))
            except (UnicodeDecodeError, fl.FrameLogError) as e:
                problems.append(Problem('log_unreadable', f"log.csv can't be read "
                                        f"({e}); rebuild it from the frames.", True))
            else:
                if session.log_sha256 is None:
                    problems.append(Problem(
                        'log_unfinished', "Logging did not finish (the app closed "
                        "while logging). The frames logged up to then are kept.", False))
                elif sha256_file(lp) != session.log_sha256:
                    problems.append(Problem(
                        'log_modified', "log.csv was changed outside the app (for "
                        "example re-saved by Excel), so its numbers may no longer "
                        "match the frames. Rebuild it to be sure.", True))
        if log is None:
            log = fl.FrameLog(session.columns)

        folder = cls(path, session, log, problems)
        missing = [r['frame_index'] for r in log.rows if r['frame_index'] not in folder._frames]
        if missing:
            folder.problems.append(Problem(
                'frames_missing', f"{len(missing)} logged frame(s) have no image in "
                f"frames/ (first: frame {missing[0]}); their rows stay but can't be "
                f"re-measured.", False))
        if session.calibration and not (path / CALIBRATION_FILE).is_file():
            folder.problems.append(Problem(
                'calibration_missing', "session.json names calibration.csv but the "
                "file is not in the folder; upload the calibration again.", False))
        return folder

    # -- frames ------------------------------------------------------------------
    def frame_indices(self):
        return sorted(self._frames)

    def read_frame(self, frame_index, prepared=True):
        """A stored frame as BGR; ``prepared`` also applies the threshold."""
        p = self._frames.get(int(frame_index))
        if p is None:
            raise SessionError(f"frame {frame_index} is not stored in this session")
        img = _read_png(p)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img
        if prepared:
            img = apply_prep(img, split_prep(self.session.prep_ops)[1])
        return img

    def measured_frames(self):
        """Log rows of the stored frames whose fit worked, earliest first.
        Any of them can give Yang the drop geometry."""
        return [r for r in sorted(self.log.rows, key=lambda r: (r['time_s'], r['frame_index']))
                if r['fit_ok'] and r['frame_index'] in self._frames]

    def first_measured(self, frame_index=None):
        """(row, prepared image) of frame ``frame_index``, or of the earliest
        frame whose fit worked; (None, None) if there is none.  Yang uses it
        as the drop geometry."""
        for r in self.measured_frames():
            if frame_index is None or r['frame_index'] == frame_index:
                return r, self.read_frame(r['frame_index'])
        return None, None

    # -- rebuild -----------------------------------------------------------------
    def rebuild_log(self, columns=None, prep_ops=None, progress_cb=None,
                    is_cancelled=None):
        """Re-measure every stored frame and write a new log.

        ``columns``: new header (default: keep); ``prep_ops``: new
        preparation, pixel steps only (crops are already in the stored
        frames).  Times and clock times come from the old log, or from the
        frame file names when the log is gone.  The old log is replaced
        only once every frame is done."""
        columns = list(columns) if columns is not None else list(self.session.columns)
        if prep_ops is not None:
            new_ops = _normalise_prep(prep_ops)
            if any(op == 'crop' for op, _ in new_ops):
                raise SessionError("stored frames are already cropped; a rebuild "
                                   "can only change the threshold")
            crops = split_prep(self.session.prep_ops)[0]
        try:
            new = fl.FrameLog(columns)
            need = fl.required_columns(self.session.analyses)
        except fl.FrameLogError as e:
            raise SessionError(str(e)) from None
        missing = [k for k in need if k not in columns]
        if missing:
            raise SessionError(f"the enabled analyses need column(s) {', '.join(missing)}")
        pixel = (new_ops if prep_ops is not None
                 else split_prep(self.session.prep_ops)[1])

        old = {r['frame_index']: r for r in self.log.rows}
        items = []
        for i in set(old) | set(self._frames):
            r = old.get(i)
            if r is not None:
                items.append((r['time_s'], i, r.get('wall_clock', '')))
            else:
                m = _FRAME_NAME.match(self._frames[i].name)
                items.append((float(m.group(2)), i, ''))
        items.sort()

        n = len(items)
        for k, (t, i, clock) in enumerate(items):
            if is_cancelled is not None and is_cancelled():
                raise FrameSeriesCancelled()
            if i in self._frames:
                rec = self._measure(_to_gray(self.read_frame(i, prepared=False)), pixel)
            else:
                rec = dict(fit_ok=False, qc_pass=False, error='frame image missing')
            rec.update(t_s=t, frame_index=i, wall_clock=clock)
            new.append_record(rec)
            if progress_cb is not None:
                progress_cb(k + 1, n, rec)

        self.log = new
        self.session.columns = columns
        if prep_ops is not None:
            self.session.prep_ops = crops + pixel
        _atomic_write_bytes(self.log_path,
                            new.to_text(self._header_lines()).encode('utf-8'))
        self.problems = [p for p in self.problems if not p.fixed_by_rebuild]
        self.finish()
        return new

    # -- calibration and analyses ----------------------------------------------
    def set_calibration(self, src_path):
        """Validate a gamma(c) calibration file and copy it into the folder."""
        from core.calibration import CalibrationError, parse_calibration_template
        try:
            parse_calibration_template(src_path)
        except CalibrationError as e:
            raise SessionError(f"calibration not accepted: {e}") from None
        _atomic_write_bytes(self.path / CALIBRATION_FILE, Path(src_path).read_bytes())
        self.session.calibration = CALIBRATION_FILE
        self._write_session()
        self.problems = [p for p in self.problems if p.code != 'calibration_missing']

    def clear_calibration(self):
        """Remove the gamma(c) calibration from the session (the Pendant
        window's Clear).  Stored results keep the values they were made with."""
        p = self.path / CALIBRATION_FILE
        if p.is_file():
            p.unlink()
        self.session.calibration = None
        self._write_session()
        self.problems = [q for q in self.problems if q.code != 'calibration_missing']

    def set_needle_id(self, id_mm):
        """Set the needle's inner diameter (mm; None = automatic, 70 % of the
        outer diameter).  It only shapes the diffusion model's domain, so it
        can change after logging; Yang and simulation results made with
        another value become out of date."""
        od = self.session.needle['od_mm']
        if id_mm is not None:
            id_mm = float(id_mm)
            if not (math.isfinite(id_mm) and 0 < id_mm < od):
                raise SessionError(f"the inner diameter must be above 0 and below the outer "
                                   f"diameter ({od:g} mm)")
        self.session.needle = {k: v for k, v in self.session.needle.items() if k != 'id_mm'}
        if id_mm is not None:
            self.session.needle['id_mm'] = id_mm
        self._write_session()

    def set_ift_curve(self, src_path):
        """Validate an IFT-vs-time CSV (core.ift_data) and copy it into the
        folder as the curve Yang can fit instead of the logged IFT.  Returns
        the parsed curve ({'t', 'gamma', 'warnings'})."""
        from core.ift_data import IFTDataError, load_ift_curve
        try:
            data = load_ift_curve(src_path)
        except IFTDataError as e:
            raise SessionError(f"IFT curve not accepted: {e}") from None
        _atomic_write_bytes(self.path / IFT_CURVE_FILE, Path(src_path).read_bytes())
        return data

    def set_volume_curve(self, src_path):
        """Validate a volume-vs-time CSV (core.volume_data) and copy it into the
        folder as the curve Bagalkot can fit instead of the logged volume
        (the Bagalkot window's Load volume curve).  Returns the parsed curve
        ({'t', 'volume', 'warnings'})."""
        from core.volume_data import VolumeDataError, load_volume_curve
        try:
            data = load_volume_curve(src_path)
        except VolumeDataError as e:
            raise SessionError(f"volume curve not accepted: {e}") from None
        _atomic_write_bytes(self.path / VOLUME_CURVE_FILE, Path(src_path).read_bytes())
        return data

    def volume_curve(self):
        """The stored volume curve ({'t', 'volume', 'warnings'}) or None."""
        from core.volume_data import VolumeDataError, load_volume_curve
        p = self.volume_curve_path
        if p is None:
            return None
        try:
            return load_volume_curve(p)
        except VolumeDataError as e:
            raise SessionError(f"{VOLUME_CURVE_FILE} can't be read: {e}") from None

    def ift_curve(self):
        """The stored IFT curve ({'t', 'gamma', 'warnings'}) or None."""
        from core.ift_data import IFTDataError, load_ift_curve
        p = self.ift_curve_path
        if p is None:
            return None
        try:
            return load_ift_curve(p)
        except IFTDataError as e:
            raise SessionError(f"{IFT_CURVE_FILE} can't be read: {e}") from None

    def _analysis_path(self, name):
        if not isinstance(name, str) or not _ANALYSIS_NAME.match(name):
            raise SessionError(f"bad analysis name {name!r}")
        return self.path / ANALYSES_DIR / f"{name}.json"

    def save_analysis(self, name, data):
        """Store an analysis's settings and results, stamped with the log
        they were computed from (so a later rebuild marks them out of date)."""
        p = self._analysis_path(name)
        p.parent.mkdir(exist_ok=True)
        doc = dict(name=name, saved=_now(), log_sha256=self.session.log_sha256,
                   data=_jsonable(data))
        _atomic_write_bytes(p, json.dumps(doc, indent=2, ensure_ascii=False).encode('utf-8'))

    def load_analysis(self, name):
        """The stored document ({'name', 'saved', 'log_sha256', 'data'}) or None."""
        p = self._analysis_path(name)
        if not p.is_file():
            return None
        try:
            return json.loads(p.read_text(encoding='utf-8'))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise SessionError(f"analyses/{name}.json is not valid JSON: {e}") from None

    def analysis_out_of_date(self, name):
        return bool(self.out_of_date_reason(name))

    def out_of_date_reason(self, name):
        """Why a stored result no longer matches the session's inputs, or ''."""
        doc = self.load_analysis(name)
        if doc is None:
            return ''
        settings = (doc.get('data') or {}).get('settings') or {}
        if 'needle_id_mm' in settings and \
                settings['needle_id_mm'] != self.session.needle.get('id_mm'):
            return ('The needle inner diameter changed since this ran, so the model was '
                    'built on another domain.')
        if settings.get('curve') == 'file':          # fitted to a curve from a file
            p = self.path / settings.get('curve_file', IFT_CURVE_FILE)
            if not p.is_file() or sha256_file(p) != settings.get('curve_sha256'):
                return 'The curve file changed since this ran.'
            return ''
        if doc.get('log_sha256') != self.session.log_sha256:
            return 'The frame log changed since this ran (for example it was rebuilt).'
        return ''

    # -- what can run --------------------------------------------------------------
    def availability(self):
        """{analysis: (can_run, reason)} for swelling, bagalkot, yang and simulation."""
        rows = self.log.rows
        n = len(rows)

        def series(col, label):
            if not self.log.has(col):
                return False, f"{label} was not logged; rebuild the log with it"
            k = sum(1 for r in rows if r['fit_ok'] and math.isfinite(r[col]))
            if k < MIN_SERIES_FRAMES:
                what = "a single image" if n == 1 else f"{k} measured frame(s)"
                return False, (f"needs at least {MIN_SERIES_FRAMES} frames with "
                               f"{label}; this session has {what}")
            return True, ''

        out = {'swelling': series('volume_uL', 'a volume'),
               'bagalkot': self.bagalkot_availability()}
        out['yang'] = self.yang_availability()
        out['simulation'] = ((True, '') if self.measured_frames() else
                             (False, "no stored frame with a successful fit for the drop geometry"))
        return out

    def bagalkot_availability(self, curve=None):
        """(can_run, reason) for Bagalkot with V(t) from ``curve``: 'log' (the
        logged volume), 'file' (volume_curve.csv) or None (either)."""
        why = {}
        if curve in (None, 'log'):
            rows = self.log.rows
            k = (sum(1 for r in rows if r['fit_ok'] and math.isfinite(r['volume_uL']))
                 if self.log.has('volume_uL') else 0)
            if k >= MIN_SERIES_FRAMES:
                return True, ''
            what = "a single image" if len(rows) == 1 else f"{k} measured frame(s)"
            why['log'] = (f"needs at least {MIN_SERIES_FRAMES} frames with a volume; this "
                          f"session has {what}")
        if curve in (None, 'file'):
            try:
                c = self.volume_curve()
            except SessionError as e:
                c, why['file'] = None, str(e)
            if c is not None and len(c['t']) >= MIN_SERIES_FRAMES:
                return True, ''
            if 'file' not in why:
                why['file'] = ("needs a volume curve file (CSV)" if c is None else
                               f"the volume curve file has {len(c['t'])} points; the fit "
                               f"needs at least {MIN_SERIES_FRAMES}")
        if curve is None:
            return False, (f"{why['log']}, or a volume curve file (CSV)")
        return False, why[curve]

    def yang_availability(self, curve=None):
        """(can_run, reason) for Yang with the IFT curve from ``curve``:
        'log' (the logged IFT), 'file' (ift_curve.csv) or None (either)."""
        if not self.measured_frames():
            return False, "no stored frame with a successful fit for the drop geometry"
        if not (self.calibration_path and self.calibration_path.is_file()):
            return False, "needs a gamma(c) calibration curve"
        why = {}
        if curve in (None, 'log'):
            k = (sum(1 for r in self.log.rows if r['qc_pass'] and math.isfinite(r['ift_mNm'])
                     and r['ift_mNm'] > 0) if self.log.has('ift_mNm') else 0)
            if k >= YANG_MIN_POINTS:
                return True, ''
            what = "a single image" if len(self.log.rows) == 1 else f"{k}"
            why['log'] = (f"needs at least {YANG_MIN_POINTS} logged frames whose IFT passed "
                          f"QC; this session has {what}")
        if curve in (None, 'file'):
            try:
                c = self.ift_curve()
            except SessionError as e:
                c, why['file'] = None, str(e)
            if c is not None and len(c['t']) >= YANG_MIN_POINTS:
                return True, ''
            if 'file' not in why:
                why['file'] = ("needs an IFT curve file (CSV)" if c is None else
                               f"the IFT curve file has {len(c['t'])} points; the fit needs "
                               f"at least {YANG_MIN_POINTS}")
        if curve is None:
            return False, (f"needs an IFT curve: at least {YANG_MIN_POINTS} logged frames "
                           f"whose IFT passed QC, or an IFT curve file (CSV)")
        return False, why[curve]

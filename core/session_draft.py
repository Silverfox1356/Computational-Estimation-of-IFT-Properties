"""
core.session_draft
==================

A session being set up, before anything is written to disk: the source and
its reference frame, the preparation (crop, threshold), fluids and needle,
the scale, the baseline, how frames will be sampled, and which columns to
log.  :meth:`SessionDraft.to_session` gives the :class:`core.session.Session`
the logger writes; :meth:`SessionDraft.missing` says what is still needed,
per Analysis Builder node.

The reference frame is the frame the user prepares on: the video frame at
the moment the drop formed, the image itself, or a frame captured from the
camera.  Crops are stored as preparation steps, so the baseline row is in
the coordinates of the cropped frame and is cleared whenever the crop
changes.

Nothing here touches Qt.
"""

import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from core import frame_log as fl
from core import presets as P
from core.frame_series import (analyse_frame, apply_prep, default_first_step,
                               frames_for_times, read_frames, sample_times,
                               video_info)
from core.image_processing import attempt_auto_baseline, measure_reference
from core.pendant_calcs import extract_pendant_edges
from core.session import Session, SessionError, split_prep

KINDS = ('video', 'image', 'camera')
SERIES_ANALYSES = ['swelling', 'bagalkot', 'yang']
SECONDS_PER_FRAME = 6.0         # rough fit time for the plan estimate


class DraftError(ValueError):
    """Something the user asked for can't be done; message is user-facing."""


def read_image(path):
    """BGR image from a file, also for non-ASCII paths on Windows."""
    try:
        data = np.fromfile(str(path), np.uint8)
    except OSError as e:
        raise DraftError(f"cannot read {Path(path).name}: {e.strerror or e}") from None
    img = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if img is None:
        raise DraftError(f"{Path(path).name} is not an image this app can read")
    return img


def _defaults():
    st = P.load_state()
    n, f = st['needle'], st['fluid']
    needle = dict(od_mm=float(n['od_mm']), tol_rel=float(n['od_tol_pct']) / 100.0,
                  name=st.get('needle_name') or 'custom needle')
    fluids = dict(density_inner=float(f['rho_in']), density_outer=float(f['rho_out']),
                  rho_in_tol_rel=float(f['rho_in_tol_pct']) / 100.0,
                  rho_out_tol_rel=float(f['rho_out_tol_pct']) / 100.0,
                  name=st.get('fluid_name') or 'custom fluids',
                  pressure_MPa=f.get('P_MPa'), temperature_C=f.get('T_C'))
    return needle, fluids, float(st.get('ts_interval', 1.0))


@dataclass
class SessionDraft:
    kind: str
    source: dict
    reference: np.ndarray
    prep_ops: list = field(default_factory=list)
    needle: dict = None
    fluids: dict = None
    scale: dict = None             # pixel_to_m, method, reference, width_px
    baseline: dict = None          # y_px, method
    sampling: dict = None
    columns: list = None           # chosen optional columns
    analyses: list = None
    calibration_path: str = None   # copied into the folder when logging starts
    ift_curve_path: str = None     # single frame: IFT(t) for Yang, copied in too
    preview: dict = None           # last measure_reference() record

    def __post_init__(self):
        if self.kind not in KINDS:
            raise DraftError(f"unknown source kind {self.kind!r}")
        needle, fluids, interval = _defaults()
        self.needle = self.needle or needle
        self.fluids = self.fluids or fluids
        if self.sampling is None:
            if self.kind == 'video':
                self.sampling = dict(mode='dense_early', n_frames=120, first_step_s=None,
                                     interval_s=5.0, t_start_s=0.0, t_end_s=None)
            elif self.kind == 'camera':
                self.sampling = dict(interval_s=interval, duration_s=None)
            else:
                self.sampling = {}
        if self.analyses is None:
            self.analyses = [] if self.kind == 'image' else list(SERIES_ANALYSES)
        if self.columns is None:
            self.columns = list(fl.OPTIONAL_KEYS)

    # -- construction ---------------------------------------------------------
    @classmethod
    def from_video(cls, path):
        try:
            info = video_info(path)
        except IOError as e:
            raise DraftError(str(e)) from None
        frames = list(read_frames(path, [0]))
        if not frames:
            raise DraftError("the video has no readable first frame")
        src = dict(name=Path(path).name, path=str(path), fps=info['fps'],
                   n_frames=info['n_frames'], width=info['width'],
                   height=info['height'], duration_s=info['duration'], t_start_s=0.0)
        return cls('video', src, frames[0][1])

    @classmethod
    def from_image(cls, path):
        img = read_image(path)
        h, w = img.shape[:2]
        return cls('image', dict(name=Path(path).name, path=str(path), width=w, height=h), img)

    @classmethod
    def from_camera(cls, frame, camera_type):
        h, w = frame.shape[:2]
        return cls('camera', dict(name=f"{camera_type} camera", camera_type=camera_type,
                                  width=w, height=h), frame.copy())

    # -- frames ---------------------------------------------------------------
    def stored_frame(self, raw=None):
        """The frame as it will be stored: crops applied, no threshold."""
        crops = split_prep(self.prep_ops)[0]
        return apply_prep(self.reference if raw is None else raw, crops)

    def prepared_frame(self, raw=None):
        """The frame as the fit sees it: crops and threshold applied."""
        return apply_prep(self.reference if raw is None else raw, self.prep_ops)

    def set_reference(self, frame):
        """New reference frame (camera capture); the preparation stays."""
        self.reference = frame.copy()
        self.preview = None

    def set_t_start(self, t):
        """Video: the moment the drop formed; its frame becomes the reference."""
        if self.kind != 'video':
            raise DraftError("only a video has a start time")
        t = float(t)
        dur = self.source['duration_s']
        if not 0 <= t < dur:
            raise DraftError(f"the start time must be between 0 and {dur:.2f} s")
        idx = min(int(round(t * self.source['fps'])), self.source['n_frames'] - 1)
        frames = list(read_frames(self.source['path'], [idx]))
        if not frames:
            raise DraftError(f"frame {idx} can't be read from the video")
        self.reference = frames[0][1]
        self.sampling['t_start_s'] = t
        self.source['t_start_s'] = t
        self.preview = None

    # -- preparation ------------------------------------------------------------
    def set_crop(self, x, y, w, h):
        """Crop the stored frame further (x, y, w, h in its coordinates)."""
        H, W = self.stored_frame().shape[:2]
        x, y = max(0, int(x)), max(0, int(y))
        w, h = min(int(w), W - x), min(int(h), H - y)
        if w < 10 or h < 10:
            raise DraftError("the crop is smaller than 10 x 10 px")
        self.prep_ops = [op for op in self.prep_ops if op[0] == 'crop'] + \
            [('crop', (x, y, w, h))] + [op for op in self.prep_ops if op[0] != 'crop']
        self.baseline = None            # its row was in the old coordinates
        self.preview = None

    def clear_crop(self):
        self.prep_ops = [op for op in self.prep_ops if op[0] != 'crop']
        self.baseline = None
        self.preview = None

    def crop_box(self):
        """The overall crop (x, y, w, h) in reference-frame coordinates, or None."""
        x0 = y0 = 0
        box = None
        for op, (x, y, w, h) in (o for o in self.prep_ops if o[0] == 'crop'):
            x0, y0 = x0 + x, y0 + y
            box = (x0, y0, w, h)
        return box

    def threshold(self):
        levels = [a for op, a in self.prep_ops if op == 'threshold']
        return levels[-1] if levels else None

    def set_threshold(self, level):
        """A fixed level 0-255, or None for automatic (Otsu) per frame."""
        ops = [op for op in self.prep_ops if op[0] != 'threshold']
        if level is not None:
            level = int(level)
            if not 0 <= level <= 255:
                raise DraftError("the threshold must be between 0 and 255")
            ops.append(('threshold', level))
        self.prep_ops = ops
        self.preview = None

    # -- fluids, needle ------------------------------------------------------------
    def set_ift_curve(self, path):
        """Single frame: the IFT-vs-time CSV Yang's fit will use (validated
        now, copied into the session folder when it is saved).  Returns the
        parsed curve; None clears it."""
        if path is None:
            self.ift_curve_path = None
            return None
        from core.ift_data import IFTDataError, load_ift_curve
        try:
            c = load_ift_curve(path)
        except IFTDataError as e:
            raise DraftError(f"IFT curve not accepted: {e}") from None
        self.ift_curve_path = str(path)
        return c

    def set_needle(self, od_mm, tol_rel=None, name=None, id_mm=...):
        """``id_mm``: the inner diameter, used only for the diffusion model's
        domain; None = automatic (70 % of the outer diameter); left out = keep."""
        if not (isinstance(od_mm, (int, float)) and 0 < od_mm <= 100):
            raise DraftError("the needle diameter must be between 0 and 100 mm")
        needle = dict(self.needle, od_mm=float(od_mm),
                      **({} if tol_rel is None else dict(tol_rel=float(tol_rel))),
                      **({} if name is None else dict(name=name)))
        if id_mm is not ...:
            needle.pop('id_mm', None)
            if id_mm is not None:
                needle['id_mm'] = float(id_mm)
        if needle.get('id_mm') is not None and not 0 < needle['id_mm'] < needle['od_mm']:
            raise DraftError("the inner diameter must be above 0 and below the outer diameter")
        self.needle = needle
        self.preview = None

    def set_fluids(self, **values):
        f = dict(self.fluids, **values)
        if f['density_inner'] == f['density_outer']:
            raise DraftError("the two densities are equal, so no IFT can be computed")
        for k in ('density_inner', 'density_outer'):
            if not 0 < float(f[k]) <= 30000:
                raise DraftError(f"{k.replace('_', ' ')} must be between 0 and 30000 kg/m3")
        self.fluids = f
        self.preview = None

    # -- scale --------------------------------------------------------------------
    def set_scale_direct(self, um_per_px):
        v = float(um_per_px)
        if not 0.001 <= v <= 10000:
            raise DraftError("the scale must be between 0.001 and 10000 um/px")
        self.scale = dict(pixel_to_m=v * 1e-6, method='entered', reference='typed in')
        self.preview = None

    def set_scale_from_length(self, length_px, length_mm, method, reference):
        length_px, length_mm = float(length_px), float(length_mm)
        if length_px < 5:
            raise DraftError("the reference is shorter than 5 px; draw it again")
        if not 0 < length_mm <= 1000:
            raise DraftError("the reference size must be between 0 and 1000 mm")
        self.scale = dict(pixel_to_m=length_mm / 1000.0 / length_px, method=method,
                          reference=reference, width_px=length_px)
        self.preview = None

    def measure_box(self, x, y, w, h, mode='needle'):
        """Size in px of the needle (or sphere) inside a box drawn on the
        stored frame, and the marker to draw.  Raises DraftError."""
        img = self.stored_frame()
        H, W = img.shape[:2]
        x0, y0 = max(0, int(x)), max(0, int(y))
        x1, y1 = min(W, int(x + w)), min(H, int(y + h))
        if x1 - x0 < 5 or y1 - y0 < 5:
            raise DraftError(f"the box is too small; draw it around the {mode}")
        px, marker = measure_reference(img[y0:y1, x0:x1], mode)
        if not px:
            raise DraftError(f"no object found in the box; draw it tightly around the {mode}")
        if marker[0] == 'line':
            marker = ('line', (marker[1][0] + x0, marker[1][1] + y0),
                      (marker[2][0] + x0, marker[2][1] + y0))
        else:
            marker = ('circle', (marker[1][0] + x0, marker[1][1] + y0), marker[2])
        return px, marker

    # -- baseline ----------------------------------------------------------------
    def auto_baseline(self):
        edges = extract_pendant_edges(self.prepared_frame())
        if edges is None or len(edges) < 20:
            raise DraftError("no drop outline found; check the crop and threshold")
        width = (self.scale or {}).get('width_px')
        if not width:
            top = (edges['right_x'] - edges['left_x']).to_numpy()[:10]
            width = float(np.median(top)) if len(top) else 0.0
        y = attempt_auto_baseline(edges, width) if width > 0 else None
        if y is None:
            raise DraftError("the needle tip was not found automatically; click on it instead")
        self.set_baseline(y, 'auto')
        return y

    def set_baseline(self, y, method='manual'):
        H = self.stored_frame().shape[0]
        if not 0 <= y < H:
            raise DraftError("the baseline must be inside the frame")
        self.baseline = dict(y_px=float(y), method=method)
        self.preview = None

    # -- sampling -----------------------------------------------------------------
    def sample_plan(self):
        """Video: dict(times, frames, t_last, min_step, max_step); camera:
        dict(interval_s, n_expected); image: dict(n=1).  Raises DraftError."""
        s = self.sampling
        if self.kind == 'image':
            return dict(n=1)
        if self.kind == 'camera':
            iv = float(s.get('interval_s') or 0)
            if not iv > 0:
                raise DraftError("the camera interval must be more than 0 s")
            dur = s.get('duration_s')
            n = int(dur // iv) + 1 if dur else None
            return dict(n=n, interval_s=iv, duration_s=dur)
        src = self.source
        t0 = float(s.get('t_start_s', 0.0))
        t1 = s.get('t_end_s')
        t1 = src['duration_s'] if t1 in (None, '') else min(float(t1), src['duration_s'])
        if not 0 <= t0 < t1:
            raise DraftError("'Analyse until' must be after the start and inside the video")
        span = t1 - t0
        try:
            if s.get('mode', 'dense_early') == 'dense_early':
                n = int(s.get('n_frames') or 0)
                first = s.get('first_step_s') or default_first_step(span, src['fps'])
                times = sample_times(span, 'dense_early', n_frames=n, first_step=float(first))
            else:
                times = sample_times(span, 'interval', interval=float(s.get('interval_s') or 0))
        except ValueError as e:
            raise DraftError(str(e)) from None
        idx, t_act = frames_for_times(times, src['fps'], t0, src['n_frames'])
        if not len(idx):
            raise DraftError("no frames fall inside the chosen times")
        steps = np.diff(t_act) if len(t_act) > 1 else np.array([0.0])
        return dict(n=len(idx), times=times, frames=idx, t_last=float(t_act[-1]),
                    min_step=float(steps.min()), max_step=float(steps.max()))

    # -- what is missing -------------------------------------------------------------
    def missing(self):
        """[(node key, message)] in tree order; empty when logging can start."""
        out = []
        if self.scale is None:
            out.append(('scale', 'Calibrate the scale (needle width or a known length).'))
        if self.baseline is None:
            out.append(('baseline', 'Set the baseline at the needle tip.'))
        try:
            self.sample_plan()
        except DraftError as e:
            out.append(('logger', f"Check the sampling: {e}"))
        return out

    def column_keys(self):
        return fl.column_keys(self.columns, analyses=self.analyses,
                              camera=self.kind == 'camera')

    def to_session(self):
        if self.missing():
            raise DraftError(self.missing()[0][1])
        src = {k: v for k, v in self.source.items() if k != 'duration_s'}
        src['kind'] = self.kind
        sampling = dict(self.sampling)
        if self.kind == 'video':
            plan = self.sample_plan()
            sampling.update(planned_frames=plan['n'])
        try:
            return Session(source=src, scale={k: v for k, v in self.scale.items()},
                           baseline=dict(self.baseline), needle=dict(self.needle),
                           fluids=dict(self.fluids), prep_ops=list(self.prep_ops),
                           sampling=sampling, columns=self.column_keys(),
                           analyses=list(self.analyses))
        except SessionError as e:
            raise DraftError(str(e)) from None

    def analysis_params(self):
        if self.scale is None or self.baseline is None:
            raise DraftError("calibrate the scale and set the baseline first")
        return Session(source=dict(kind=self.kind), scale=self.scale, baseline=self.baseline,
                       needle=self.needle, fluids=self.fluids).analysis_params()

    def measure_reference(self):
        """Fit the reference frame (what the old Static mode did)."""
        rec = analyse_frame(self.prepared_frame(), self.analysis_params())
        self.preview = rec
        return rec

"""
core.frame_series
=================

Analyse a drop video at chosen times, not every frame (a 10-min recording
at 60 fps has 36,000 frames).  For each sampled frame: apply the same
preparation the user set on the first frame (crop, threshold), extract the
contour and fit it, and keep a record of IFT, volume, area and QC.

The resulting table feeds both routes after the swelling check
(``core.swelling``):

- Yang: the first frame is the domain and IFT(t) is the data;
- Bagalkot: V(t) is the data.

Sampling modes
--------------
``'dense_early'``  more frames at the start, where IFT and volume change
                   fastest: t = 0 plus geometrically spaced times from
                   ``first_step`` to the end.
``'interval'``     one frame every ``interval`` seconds.

Times are seconds after ``t_start`` (the moment the drop is formed in the
video; frames before it are ignored).  Each requested time is snapped to
the nearest frame, and duplicates are removed, so the actual count can be
lower than requested when the first steps are shorter than a frame.

Nothing here touches Qt; the GUI wraps :func:`analyse_video` in a worker
thread and passes a progress callback and a cancel check.
"""

import csv
from pathlib import Path

import cv2
import numpy as np

from core.image_processing import binarize_drop_image
from core.pendant_calcs import extract_pendant_edges
from core.physics_calcs import calculate_physics, compute_worthington

SAMPLING_MODES = ('dense_early', 'interval')
MAX_SAMPLED_FRAMES = 5000          # guard against a typo like interval=0.001


class FrameSeriesCancelled(Exception):
    """Raised inside :func:`analyse_video` when the cancel check fires."""


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def default_first_step(span, fps):
    """Dense-early first step: 0.1 % of the analysed span, never shorter
    than one frame."""
    return max(1.0 / fps if fps > 0 else 0.0, 1e-3 * span)


def sample_times(span, mode='dense_early', n_frames=120, first_step=None,
                 interval=5.0):
    """Requested sample times (s after t_start), before snapping to frames.

    span        : analysed duration in seconds (> 0)
    dense_early : ``n_frames`` times: 0, then geometric from ``first_step``
                  to ``span``
    interval    : 0, interval, 2·interval, ... ≤ span
    """
    if span <= 0:
        raise ValueError("the analysed part of the video has zero length")
    if mode == 'dense_early':
        n = int(n_frames)
        if n < 3:
            raise ValueError("dense-early sampling needs at least 3 frames")
        if n > MAX_SAMPLED_FRAMES:
            raise ValueError(f"at most {MAX_SAMPLED_FRAMES} frames")
        dt1 = float(first_step) if first_step else 1e-3 * span
        if not 0 < dt1 < span:
            raise ValueError("first step must be > 0 and shorter than the span")
        return np.concatenate([[0.0], np.geomspace(dt1, span, n - 1)])
    if mode == 'interval':
        step = float(interval)
        if step <= 0:
            raise ValueError("interval must be > 0 s")
        n = int(np.floor(span / step + 1e-9)) + 1
        if n > MAX_SAMPLED_FRAMES:
            raise ValueError(
                f"interval {step:g} s gives {n} frames (max "
                f"{MAX_SAMPLED_FRAMES}); use a longer interval")
        return np.arange(n) * step
    raise ValueError(f"unknown sampling mode {mode!r}")


def frames_for_times(times, fps, t_start, n_total):
    """Snap times (s after t_start) to unique frame indices inside the video.
    Returns (indices, times_actual) with times_actual = index/fps − t_start."""
    idx = np.rint((np.asarray(times, float) + t_start) * fps).astype(int)
    idx = np.unique(np.clip(idx, 0, max(0, n_total - 1)))
    return idx, idx / fps - t_start


# ---------------------------------------------------------------------------
# Video access
# ---------------------------------------------------------------------------
def video_info(path):
    """{'fps', 'n_frames', 'duration', 'width', 'height'} of a video file."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"cannot open video {path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    if fps <= 0 or n <= 0:
        raise IOError("the video reports no frame rate or frame count")
    return dict(fps=fps, n_frames=n, duration=n / fps, width=w, height=h)


def read_frames(path, indices):
    """Yield (index, frame) for the sorted frame indices.

    Short gaps are walked with grab() (no full decode of skipped frames);
    long gaps seek directly."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"cannot open video {path}")
    try:
        pos = 0
        for i in sorted(int(k) for k in indices):
            if i - pos > 30 or i < pos:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                pos = i
            while pos < i:
                if not cap.grab():
                    return
                pos += 1
            ok, frame = cap.read()
            pos += 1
            if not ok:
                return
            yield i, frame
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# Per-frame analysis
# ---------------------------------------------------------------------------
def apply_prep(frame, prep_ops):
    """Apply the preparation the user made on the first frame, in order.
    ``prep_ops`` items: ('crop', (x, y, w, h)) or ('threshold', value)."""
    img = frame
    for op, arg in prep_ops or ():
        if op == 'crop':
            x, y, w, h = arg
            H, W = img.shape[:2]
            img = img[max(0, y):min(H, y + h), max(0, x):min(W, x + w)]
        elif op == 'threshold':
            img = binarize_drop_image(img, int(arg))
        else:
            raise ValueError(f"unknown preparation step {op!r}")
    return img


def surface_area_mm2(results, pixel_to_m):
    """Area of the drop surface below the baseline, from the smoothed
    per-row radii (sum of conical frusta)."""
    ys = np.asarray(results['ys_use'], float)
    r = 0.5 * (np.asarray(results['rxs'], float) - np.asarray(results['lxs'], float))
    if len(ys) < 2:
        return float('nan')
    slant = np.hypot(np.diff(ys), np.diff(r))
    area_px2 = float(np.sum(np.pi * (r[:-1] + r[1:]) * slant))
    return area_px2 * (pixel_to_m * 1e3) ** 2


def analyse_frame(frame, params, warm_start=None):
    """Measure one prepared frame.  ``params``: baseline_y, pixel_to_m,
    density_inner, density_outer, needle_tol_rel, rho_in_tol_rel,
    rho_out_tol_rel, needle_diameter_m.

    ``warm_start`` (previous frame's beta, R0_px) speeds up the fit.

    Returns a record dict; 'results' holds the full physics output (kept
    so the first frame can become the Step 1 image for the Yang route)."""
    rec = dict(fit_ok=False, qc_pass=False, ift_mNm=np.nan,
               ift_unc_mNm=np.nan, volume_uL=np.nan, area_mm2=np.nan,
               R0_mm=np.nan, beta=np.nan, error='', results=None)
    edges = extract_pendant_edges(frame)
    if edges is None:
        rec['error'] = 'no drop contour found'
        return rec
    res = calculate_physics(
        edges, params['baseline_y'], params['pixel_to_m'],
        density_inner=params['density_inner'],
        density_outer=params['density_outer'],
        needle_tol_rel=params.get('needle_tol_rel', 0.01),
        density_inner_tol_rel=params.get('rho_in_tol_rel', 0.0005),
        density_outer_tol_rel=params.get('rho_out_tol_rel', 0.02),
        warm_start=warm_start)
    rec['volume_uL'] = float(res.get('volume_ul', np.nan))
    if not res.get('fit_success'):
        rec['error'] = res.get('error', 'fit failed')
        return rec
    res = compute_worthington(res, params['needle_diameter_m'])
    rec.update(fit_ok=True, qc_pass=bool(res.get('qc_pass', False)),
               ift_mNm=float(res['ift_mNm']),
               ift_unc_mNm=float(res.get('ift_uncertainty_mNm', np.nan)),
               R0_mm=float(res['R0_mm']), beta=float(res['beta']),
               area_mm2=surface_area_mm2(res, params['pixel_to_m']),
               results=res, edges=edges)
    return rec


def analyse_video(path, sample_t, t_start, prep_ops, params,
                  progress_cb=None, is_cancelled=None):
    """Analyse the frames nearest to ``sample_t`` (s after ``t_start``).

    Returns a list of records sorted by time; each has 't_s' and
    'frame_index'.  Records up to and including the first measured frame
    also carry the prepared frame under 'image' (for the Yang hand-off).  ``progress_cb(i, n, rec)`` is
    called after every frame; ``is_cancelled()`` is polled between frames.
    """
    info = video_info(path)
    idx, t_act = frames_for_times(sample_t, info['fps'], t_start, info['n_frames'])
    keep = t_act >= -0.5 / info['fps']
    idx, t_act = idx[keep], np.maximum(t_act[keep], 0.0)
    t_of = dict(zip(idx.tolist(), t_act.tolist()))
    records = []
    n = len(idx)
    warm = None
    for k, (i, frame) in enumerate(read_frames(path, idx)):
        if is_cancelled is not None and is_cancelled():
            raise FrameSeriesCancelled()
        img = apply_prep(frame, prep_ops)
        rec = analyse_frame(img, params, warm_start=warm)
        if rec['fit_ok']:
            warm = dict(beta=rec['results']['beta'],
                        R0_px=rec['results']['R0_px'])
        rec.update(t_s=float(t_of[i]), frame_index=int(i))
        if not any(r['fit_ok'] for r in records):
            rec['image'] = img          # up to the first measured frame
        records.append(rec)
        if progress_cb is not None:
            progress_cb(k + 1, n, rec)
    if len(records) < n:
        # The container ended before its reported frame count; say so.
        for rec in records[-1:]:
            rec['error'] = (rec['error'] + '; ' if rec['error'] else '') + \
                f"video ended early ({len(records)} of {n} frames read)"
    return records


# ---------------------------------------------------------------------------
# Table export / import
# ---------------------------------------------------------------------------
TABLE_COLUMNS = ('time_s', 'frame_index', 'volume_uL', 'area_mm2', 'ift_mNm',
                 'ift_unc_mNm', 'R0_mm', 'beta', 'fit_ok', 'qc_pass', 'error')


def save_frame_table(path, records, header_lines=()):
    """Write the per-frame table as CSV (``#`` lines first, then a header)."""
    with open(path, 'w', newline='', encoding='utf-8') as f:
        for line in header_lines:
            f.write(f"# {line}\n")
        w = csv.writer(f)
        w.writerow(TABLE_COLUMNS)
        for r in records:
            w.writerow([f"{r['t_s']:.4f}", r.get('frame_index', ''),
                        _fmt(r['volume_uL']), _fmt(r['area_mm2']),
                        _fmt(r['ift_mNm']), _fmt(r['ift_unc_mNm']),
                        _fmt(r['R0_mm']), _fmt(r['beta']),
                        int(bool(r['fit_ok'])), int(bool(r['qc_pass'])),
                        r.get('error', '')])


def _fmt(v):
    return '' if v is None or not np.isfinite(v) else f"{v:.6g}"


def table_arrays(records):
    """Columns of the records as numpy arrays (NaN where missing)."""
    get = lambda k: np.array([r.get(k, np.nan) for r in records], dtype=float)
    return dict(t=get('t_s'), volume=get('volume_uL'), area=get('area_mm2'),
                ift=get('ift_mNm'), ift_unc=get('ift_unc_mNm'),
                fit_ok=np.array([bool(r['fit_ok']) for r in records]),
                qc=np.array([bool(r['qc_pass']) for r in records]))


def load_frame_table_rows(path):
    """Read a table written by :func:`save_frame_table` back as a list of
    dicts (numbers as float, NaN for blanks; fit_ok / qc_pass as bool)."""
    rows = []
    with open(path, newline='', encoding='utf-8-sig') as f:
        lines = [ln for ln in f if not ln.startswith('#')]
    reader = csv.DictReader(lines)
    for r in reader:
        row = {}
        for k, v in r.items():
            if k in ('fit_ok', 'qc_pass'):
                row[k] = v.strip() == '1'
            elif k == 'error':
                row[k] = v
            else:
                row[k] = float(v) if v.strip() else float('nan')
        rows.append(row)
    return rows

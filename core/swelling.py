"""
core.swelling
=============

Decide from a measured volume curve V(t) whether the drop swells enough
that a single starting picture (Yang et al. 2006) is no longer a valid
model domain, so that Bagalkot's (2019) volume method should be used.

Swelling
--------
    S = V_end / V0 − 1

V0 is the first valid frame (Bagalkot: "the initial volume (t = 0) ...
determined experimentally"); V_end is the median of the last 10 % of
valid frames (at least 3), so one bad frame cannot decide the verdict.

Noise
-----
Frame-to-frame volume scatter σ_V is estimated from second differences of
V (robust MAD), which removes the smooth swelling trend.  The swelling is
"not measurable" when |S| < ``noise_factor`` × its own standard error.

Verdict (threshold is user-editable, default 3 %)
-------------------------------------------------
    not measurable          → Yang recommended (no usable volume signal)
    |S| < threshold         → negligible  → Yang
    S ≥ threshold           → significant → Bagalkot
    S ≤ −threshold          → the drop shrinks; neither method models it

Bagalkot's Table 4.3 (CW–n-decane) gives the error in D from treating a
swelling drop as static: 2.0 % at V_f/V0 = 1.031, 14.8 % at 1.155 and
36.0 % at 2.134.  The interpolated value is reported as an indication;
it is for his system, not a prediction for every liquid.
"""

import numpy as np

DEFAULT_THRESHOLD = 0.03
DEFAULT_NOISE_FACTOR = 3.0
TAIL_FRACTION = 0.10
MIN_TAIL = 3
MIN_FRAMES = 5

# Bagalkot (2019) Table 4.3: swelling S = V_f/V0 − 1 → % change in D,
# static vs moving interface (CW–n-decane, 25 °C).
_TABLE_43_S = np.array([0.0, 0.031, 0.155, 1.134])
_TABLE_43_DERR = np.array([0.0, 2.015, 14.812, 36.045])


def static_error_indication(S):
    """Bagalkot Table 4.3, linearly interpolated; None beyond his range."""
    S = abs(float(S))
    if S > _TABLE_43_S[-1]:
        return None
    return float(np.interp(S, _TABLE_43_S, _TABLE_43_DERR))


def volume_noise(V):
    """Robust frame-to-frame standard deviation of V (same units as V).

    For white noise of std σ, the second difference has std σ·√6."""
    V = np.asarray(V, float)
    if len(V) < 4:
        return float('nan')
    d2 = np.diff(V, 2)
    mad = np.median(np.abs(d2 - np.median(d2)))
    return float(1.4826 * mad / np.sqrt(6.0))


def swelling_check(t, V, threshold=DEFAULT_THRESHOLD,
                   noise_factor=DEFAULT_NOISE_FACTOR):
    """Swelling numbers and verdict for a volume curve.

    ``t`` in s, ``V`` in any volume unit; NaN volumes (failed frames) are
    ignored.  Returns a dict: S, S_err, V0, V_end, sigma_V, n_valid,
    still_rising, static_error_pct, verdict, recommend ('yang' |
    'bagalkot' | None) and message (plain text for the GUI).
    """
    t = np.asarray(t, float)
    V = np.asarray(V, float)
    ok = np.isfinite(t) & np.isfinite(V) & (V > 0)
    t, V = t[ok], V[ok]
    order = np.argsort(t, kind='stable')
    t, V = t[order], V[order]
    n = len(V)
    out = dict(threshold=float(threshold), noise_factor=float(noise_factor),
               n_valid=n, S=float('nan'), S_err=float('nan'),
               V0=float('nan'), V_end=float('nan'), sigma_V=float('nan'),
               still_rising=False, static_error_pct=None,
               verdict='insufficient', recommend=None)
    if n < MIN_FRAMES:
        out['message'] = (f"Only {n} frame(s) have a measured volume; at "
                          f"least {MIN_FRAMES} are needed for a swelling "
                          f"check.")
        return out

    n_tail = max(MIN_TAIL, int(np.ceil(TAIL_FRACTION * n)))
    n_tail = min(n_tail, n - 1)
    V0 = float(V[0])
    V_end = float(np.median(V[-n_tail:]))
    sigma = volume_noise(V)
    S = V_end / V0 - 1.0
    # V0 is one frame (σ); V_end a median of n_tail (≈ 1.25σ/√n_tail)
    S_err = (sigma / V0) * np.sqrt(1.0 + 1.5708 / n_tail)

    # Still rising?  Compare the last tail with the tail before it.
    still = False
    if n >= 2 * n_tail + 1:
        prev = float(np.median(V[-2 * n_tail:-n_tail]))
        still = (V_end - prev) > noise_factor * sigma * np.sqrt(2 * 1.5708 / n_tail)

    out.update(S=float(S), S_err=float(S_err), V0=V0, V_end=V_end,
               sigma_V=float(sigma), still_rising=bool(still),
               static_error_pct=static_error_indication(S))

    pct, thr = 100 * S, 100 * threshold
    noise_txt = f"noise ±{100 * S_err:.2f} %"
    if not np.isfinite(S_err) or abs(S) < noise_factor * S_err:
        out.update(verdict='not_measurable', recommend='yang',
                   message=(f"Swelling {pct:+.2f} % is within the volume "
                            f"noise ({noise_txt}): not measurable. The "
                            f"volume method has no usable signal; a static "
                            f"picture (Yang) is recommended."))
    elif S <= -threshold:
        out.update(verdict='shrinking', recommend=None,
                   message=(f"The drop shrinks by {-pct:.2f} % ({noise_txt}). "
                            f"Neither method models a shrinking drop "
                            f"(evaporation or dissolution of the drop "
                            f"liquid?). Check the video before fitting."))
    elif abs(S) < threshold:
        err = out['static_error_pct']
        out.update(verdict='negligible', recommend='yang',
                   message=(f"Swelling {pct:+.2f} % ({noise_txt}) is below "
                            f"the {thr:g} % threshold: negligible. The "
                            f"starting picture can be used (Yang)."
                            + (f" Bagalkot's Table 4.3 suggests about "
                               f"{err:.1f} % error in D from ignoring it."
                               if err is not None else "")))
    else:
        err = out['static_error_pct']
        out.update(verdict='significant', recommend='bagalkot',
                   message=(f"Swelling {pct:+.2f} % ({noise_txt}) exceeds "
                            f"the {thr:g} % threshold: significant. "
                            f"Bagalkot's volume method is recommended."
                            + (f" Treating the drop as static would change "
                               f"D by about {err:.0f} % (his Table 4.3)."
                               if err is not None else
                               " This is beyond his Table 4.3 (36 % error "
                               "at 113 % swelling).")))
    if still and out['verdict'] in ('negligible', 'significant'):
        out['message'] += (" The volume is still rising at the last frame, "
                           "so the final swelling is a lower bound.")
    return out

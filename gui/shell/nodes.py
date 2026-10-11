"""
gui.shell.nodes
===============

The Analysis Builder tree: which nodes exist and what state each is in.

Kept free of Qt so the rules can be tested without a display.  A status is
one of :data:`STATUSES`; every status comes with a one-line reason that the
tree shows as a tooltip, so a greyed-out node always says why.
"""

from collections import namedtuple
from pathlib import Path

Node = namedtuple('Node', 'key parent label icon')

NODES = (
    Node('session', None, 'Session', 'project'),
    Node('global', 'session', 'Global Definitions', 'global'),
    Node('fluids', 'global', 'Fluids & Needle', 'params'),
    Node('calibration', 'global', 'IFT Calibration Curve γ(c)', 'calcurve'),
    Node('source', 'session', 'Source', 'video'),
    Node('crop', 'source', 'Crop', 'crop'),
    Node('threshold', 'source', 'Threshold', 'threshold'),
    Node('logger', 'session', 'Frame Logger', 'series'),
    Node('scale', 'logger', 'Scale Calibration', 'ruler'),
    Node('baseline', 'logger', 'Baseline', 'baseline'),
    Node('log', 'session', 'Frame Log', 'table'),
    Node('frame_ift', 'log', 'Single-Frame IFT', 'drop'),
    Node('analyses', 'session', 'Analyses', 'compute'),
    Node('swelling', 'analyses', 'Swelling Check', 'swell'),
    Node('yang', 'analyses', 'Diffusivity (Yang)', 'inv'),
    Node('yang_model', 'yang', 'Model: Geometry, Mesh, FEM', 'mesh'),
    Node('bagalkot', 'analyses', 'Diffusivity (Bagalkot)', 'inv'),
    Node('results', 'session', 'Results', 'plot'),
    Node('report', 'results', 'Report', 'report'),
)
NODE = {n.key: n for n in NODES}

# The order the Next button walks through: the work flow, which is not quite
# the tree order (the scale and baseline come before Start logging).
NEXT_ORDER = ('fluids', 'calibration', 'crop', 'threshold', 'scale', 'baseline', 'logger',
              'log', 'frame_ift', 'swelling', 'yang', 'yang_model', 'bagalkot', 'results',
              'report')

# complete, ready to run, needs input, warning, not available, out of date
STATUSES = ('done', 'ready', 'input', 'warn', 'off', 'stale')
STATUS_LABELS = dict(done='Complete', ready='Ready', input='Needs input',
                     warn='Warning', off='Not available', stale='Out of date')


def children(key):
    return [n for n in NODES if n.parent == key]


def draft_statuses(draft):
    """{node key: (status, reason)} for a session being set up
    (core.session_draft.SessionDraft); nothing is on disk yet."""
    missing = dict(draft.missing())
    off = 'Available once the frames are logged.'
    st = {n.key: ('off', off) for n in NODES}
    f, n = draft.fluids, draft.needle
    st['fluids'] = ('done', f"{f.get('name') or 'Fluids'}: inner {f['density_inner']:g}, "
                    f"outer {f['density_outer']:g} kg/m3; needle {n['od_mm']:g} mm")
    st['calibration'] = (('done', 'Will be copied into the session.') if draft.calibration_path
                         else ('ready', 'Optional: only Diffusivity (Yang) needs it.'))
    st['global'] = _worst(st, 'global')
    st['source'] = ('done', f"{draft.kind.capitalize()}: {draft.source.get('name', '')}")
    box = draft.crop_box()
    st['crop'] = ('done', f"Cropped to {box[2]} x {box[3]} px" if box
                  else 'No crop: the whole frame is used.')
    thr = draft.threshold()
    st['threshold'] = ('done', f"Threshold level {thr}" if thr is not None
                       else 'Automatic (Otsu) per frame.')
    for key in ('scale', 'baseline'):
        st[key] = ('input', missing[key]) if key in missing else ('done', '')
    if 'scale' not in missing:
        st['scale'] = ('done', f"{draft.scale['pixel_to_m'] * 1e6:.4g} um/px "
                       f"({draft.scale.get('method')})")
    if 'baseline' not in missing:
        st['baseline'] = ('done', f"y = {draft.baseline['y_px']:g} px "
                          f"({draft.baseline.get('method')})")
    if 'logger' in missing:
        st['logger'] = ('input', missing['logger'])
    elif missing:
        st['logger'] = ('input', 'Complete the steps above, then start logging.')
    else:
        st['logger'] = ('ready', 'Ready to start logging.')
    st['log'] = ('off', 'Nothing logged yet.')
    if 'scale' in missing or 'baseline' in missing:
        st['frame_ift'] = ('off', 'Set the scale and the baseline first.')
    elif draft.preview and draft.preview.get('fit_ok'):
        st['frame_ift'] = ('done', f"Reference frame: IFT {draft.preview['ift_mNm']:.4g} mN/m.")
    else:
        st['frame_ift'] = ('ready', 'Measure the IFT of the reference frame.')
    st['session'] = (('input', f"New session, not saved yet: {len(missing)} item(s) needed.")
                     if missing else ('ready', 'New session, ready to log.'))
    return st


def node_statuses(folder):
    """{node key: (status, reason)} for an open SessionFolder, or for no
    session when ``folder`` is None."""
    if folder is None:
        out = {n.key: ('off', 'No session is open.') for n in NODES}
        out['session'] = ('input', 'Open a session folder to begin.')
        return out

    s = folder.session
    log = folder.log
    codes = {p.code: p for p in folder.problems}
    st = {}

    st['fluids'] = ('done', f"{s.fluids.get('name') or 'Fluids'}: inner "
                    f"{s.fluids['density_inner']:g} kg/m3, outer "
                    f"{s.fluids['density_outer']:g} kg/m3; needle "
                    f"{s.needle['od_mm']:g} mm")
    if 'calibration_missing' in codes:
        st['calibration'] = ('warn', codes['calibration_missing'].message)
    elif folder.calibration_path is not None:
        st['calibration'] = ('done', 'Calibration curve stored in the session.')
    else:
        st['calibration'] = ('ready', 'Optional: only Diffusivity (Yang) needs it.')
    st['global'] = _worst(st, 'global')

    src = s.source
    name = src.get('name') or src.get('path') or src.get('kind')
    if src.get('kind') == 'video' and src.get('path') and not Path(src['path']).is_file():
        st['source'] = ('warn', f"The video is no longer at {src['path']}; the "
                        f"stored frames are still usable.")
    else:
        st['source'] = ('done', f"{src.get('kind', '').capitalize()}: {name}")
    crops = [a for op, a in s.prep_ops if op == 'crop']
    thr = [a for op, a in s.prep_ops if op == 'threshold']
    st['crop'] = ('done', f"Cropped to {crops[-1][2]} x {crops[-1][3]} px" if crops
                  else 'No crop: the whole frame is used.')
    st['threshold'] = ('done', f"Threshold level {thr[-1]}" if thr
                       else 'No fixed threshold: automatic (Otsu) per frame.')

    st['scale'] = ('done', f"{s.scale['pixel_to_m'] * 1e6:.4g} um/px "
                   f"({s.scale.get('method', 'unknown method')})")
    st['baseline'] = ('done', f"y = {s.baseline['y_px']:g} px "
                      f"({s.baseline.get('method', 'unknown method')})")
    if 'log_unfinished' in codes:
        st['logger'] = ('warn', codes['log_unfinished'].message)
    elif len(log) == 0 and not folder.frame_indices():
        st['logger'] = ('input', 'No frames logged yet.')
    else:
        st['logger'] = ('done', f"{len(log)} frame(s) logged.")

    log_problems = [p for p in folder.problems
                    if p.code not in ('calibration_missing', 'log_unfinished')]
    if log_problems:
        st['log'] = ('warn', log_problems[0].message)
    else:
        n_ok = sum(1 for r in log.rows if r['fit_ok'])
        st['log'] = ('done', f"{len(log)} row(s), {n_ok} measured.")
    st['frame_ift'] = (('ready', 'Fit one logged frame again, or an image taken with the '
                        'same set-up.') if folder.frame_indices()
                       else ('off', 'No frames are stored in this session.'))

    avail = folder.availability()
    for key in ('swelling', 'yang', 'bagalkot'):
        ok, why = avail[key]
        if not ok:
            st[key] = ('off', why)
        elif folder.load_analysis(key) is not None:
            if folder.analysis_out_of_date(key):
                st[key] = ('stale', 'The log changed since this analysis ran; run it again.')
            else:
                caveat = result_caveat(key, folder.load_analysis(key))
                st[key] = ('warn', caveat) if caveat else ('done', 'Results stored in the session.')
        else:
            st[key] = ('ready', 'Ready to run.')
    st['yang_model'] = (('off', 'Available once Diffusivity (Yang) can run.')
                        if st['yang'][0] == 'off'
                        else ('ready', 'Built from the first measured frame.'))
    if all(st[k][0] == 'off' for k in ('swelling', 'yang', 'bagalkot')):
        st['analyses'] = ('off', 'No analysis can run on this session yet.')
    else:
        st['analyses'] = _worst(st, 'analyses')

    stored = [k for k in ('swelling', 'bagalkot', 'yang') if st[k][0] in ('done', 'stale', 'warn')]
    if stored:
        stale = [k for k in stored if st[k][0] == 'stale']
        st['results'] = (('stale', 'Some results are out of date.') if stale
                         else ('done', f"{len(stored)} analysis result(s) stored."))
        if 'yang' in stored:
            st['yang_model'] = ('done', 'Built when Diffusivity (Yang) ran.')
    else:
        st['results'] = ('off', 'Run an analysis first.')
    st['report'] = ('off', 'The printable report is the next step of the overhaul.')
    worst = _worst(st, None, keys=[k for k in st])
    st['session'] = (worst[0] if worst[0] in ('warn', 'input') else 'done',
                     worst[1] if worst[0] in ('warn', 'input') else
                     f"{len(log)} frame(s) logged.")
    return st


def result_caveat(key, doc):
    """Why a stored result should not be trusted as it stands, or ''."""
    try:
        r = doc['data']['result']
    except (KeyError, TypeError):
        return 'The stored result cannot be read.'
    if key == 'yang':
        if not r.get('success', True):
            return 'The fit did not converge: ' + (r.get('message') or 'see the page.')
        if not r.get('D_well_determined', True):
            return 'D is not well determined by this curve (see the page).'
    if key == 'bagalkot' and r.get('at_edge'):
        return 'The best D is at the edge of the search range; widen it.'
    return ''


_ORDER = ('warn', 'input', 'stale', 'ready', 'done', 'off')


def _worst(st, parent, keys=None):
    keys = keys if keys is not None else [n.key for n in children(parent)]
    vals = [st[k] for k in keys if k in st and st[k][0] != 'off']
    if not vals:
        return ('off', 'Nothing here can run yet.')
    for status in _ORDER:
        for v in vals:
            if v[0] == status:
                return v if status in ('warn', 'input', 'stale') else (status, '')
    return vals[0]


def next_step(current, statuses):
    """The node after ``current`` in the work flow (NEXT_ORDER) that is not
    'off', or None at the end.  From a group node (Session, Source...) it is
    the first step after it in the tree."""
    if current in NEXT_ORDER:
        later = NEXT_ORDER[NEXT_ORDER.index(current) + 1:]
    else:
        pos = {n.key: i for i, n in enumerate(NODES)}
        later = [k for k in NEXT_ORDER if pos[k] > pos.get(current, -1)]
    for key in later:
        if statuses.get(key, ('off',))[0] != 'off':
            return key
    return None


def next_required(statuses):
    """First node (tree order) that needs input or has a warning, or None."""
    for n in NODES:
        if n.key != 'session' and statuses.get(n.key, ('off',))[0] in ('input', 'warn'):
            return n.key
    return None

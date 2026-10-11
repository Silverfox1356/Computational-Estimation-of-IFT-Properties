"""
Drop Analysis Studio shell (gui.shell): tree status rules without a
display, then the window itself on Qt's offscreen platform: opening a
session, the pages of every node, the table, the frame viewer, the log
plot, rebuilding the log as a background job, and bad folders.

Plain script (project convention — no pytest):
    conda run -n ift python tests/test_studio_shell.py      (about 40 s)
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _synthetic_video import render_drop
from core.session import LOG_FILE, Session, SessionFolder
from gui.shell import nodes

_passed = 0
_failed = 0
PX = 1e-5


def check(name, cond, detail=""):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  ok   {name}" + (f"  ({detail})" if detail else ""))
    else:
        _failed += 1
        print(f"  FAIL {name}" + (f"  ({detail})" if detail else ""))


def make_folder(path, n_frames, analyses=('swelling',)):
    s = Session(source=dict(kind='image' if n_frames == 1 else 'video', name='synthetic'),
                scale=dict(pixel_to_m=PX, method='manual'),
                baseline=dict(y_px=60, method='auto'),
                needle=dict(od_mm=0.8, tol_rel=0.01),
                fluids=dict(density_inner=1000.0, density_outer=1.2, name='water / air'),
                analyses=list(analyses))
    f = SessionFolder.create(path, s)
    for k in range(n_frames):
        f.log_frame(render_drop(0.30, 60 + 3 * k)[0], 5.0 * k, k)
    f.finish()
    return f


def main():
    tmp = Path(tempfile.mkdtemp(prefix='studio test '))
    try:
        rules(tmp)
        window(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{_passed} passed, {_failed} failed")
    sys.exit(1 if _failed else 0)


def rules(tmp):
    print("── status rules (no display) ──")
    st = nodes.node_statuses(None)
    check("every node has a status", set(st) == {n.key for n in nodes.NODES})
    check("no session: session needs input", st['session'][0] == 'input')
    check("no session: the rest unavailable",
          all(st[k][0] == 'off' for k in st if k != 'session'))
    check("statuses are known", all(v[0] in nodes.STATUSES for v in st.values()))
    check("next required with no session is none", nodes.next_required(st) is None)

    one = make_folder(tmp / 'one', 1)
    st = nodes.node_statuses(SessionFolder.open(one.path))
    check("single image: logger complete", st['logger'][0] == 'done', st['logger'][1])
    check("single image: swelling not available, with reason",
          st['swelling'][0] == 'off' and 'single image' in st['swelling'][1], st['swelling'][1])
    check("single image: analyses folder not available", st['analyses'][0] == 'off')

    six = make_folder(tmp / 'six', 10)              # 10 frames: Yang's minimum
    st = nodes.node_statuses(SessionFolder.open(six.path))
    check("series: swelling ready", st['swelling'][0] == 'ready', st['swelling'][1])
    check("series: yang needs calibration",
          st['yang'][0] == 'off' and 'calibration' in st['yang'][1], st['yang'][1])
    check("series: calibration optional", st['calibration'][0] == 'ready')
    check("series: session complete", st['session'][0] == 'done', st['session'][0])
    six.save_analysis('swelling', {'settings': {'threshold': 0.03}, 'result': {'message': 'test'}})
    st = nodes.node_statuses(SessionFolder.open(six.path))
    check("stored analysis complete", st['swelling'][0] == 'done')

    lp = six.path / LOG_FILE
    lp.write_text(lp.read_text(encoding='utf-8') + '\n', encoding='utf-8')
    st = nodes.node_statuses(SessionFolder.open(six.path))
    check("edited log: warning on Frame Log", st['log'][0] == 'warn', st['log'][1])
    check("edited log: session shows the warning", st['session'][0] == 'warn')
    check("next required goes to the warning", nodes.next_required(st) == 'log')


def window(tmp):
    print("── window (offscreen) ──")
    from PyQt6.QtCore import QSettings
    from PyQt6.QtWidgets import QApplication, QMessageBox
    app = QApplication.instance() or QApplication(sys.argv)
    warnings = []
    QMessageBox.warning = staticmethod(lambda *a, **k: warnings.append(a[2] if len(a) > 2 else ''))
    QMessageBox.information = staticmethod(lambda *a, **k: None)

    from gui.shell.main_window import StudioWindow
    ini = QSettings(str(tmp / 'studio.ini'), QSettings.Format.IniFormat)
    w = StudioWindow(settings_store=ini)
    w.resize(1280, 800)
    w.show()
    app.processEvents()
    check("starts on the Start tab", w.graphics.count() == 1
          and w.graphics.tabText(0) == 'Start')
    check("no-session actions disabled", not w.a_close.isEnabled() and not w.a_rebuild.isEnabled())

    folder = make_folder(tmp / 'gui', 6)
    ok = w.open_session(str(folder.path))
    app.processEvents()
    check("session opens", ok and w.folder is not None)
    tabs = [w.graphics.tabText(i) for i in range(w.graphics.count())]
    check("frame and log plot tabs appear", tabs == ['Frame', 'Log Plot'], str(tabs))
    check("table has every row", w.table.model().rowCount() == 6)
    check("frame viewer shows an image", not w.graphics.frames.view.item.pixmap().isNull())
    check("tree status icons set", all(not it.icon(1).isNull() for it in w._items.values()))
    check("recent sessions remembered", ini.value('recent')[0] == str(folder.path))
    check("window title names the session", folder.path.name in w.windowTitle())
    mw = w.minimumSizeHint().width()
    check("fits a 1280 px screen with room to shrink", mw <= 1000, f"minimum width {mw} px")

    errors = []
    for n in nodes.NODES:
        try:
            w.select_node(n.key)
            app.processEvents()
        except Exception as e:                        # any page crash fails the test
            errors.append(f"{n.key}: {e}")
    check("every node's page builds", not errors, '; '.join(errors))

    w._show_frame(3)
    app.processEvents()
    check("table follows the shown frame", w.table.currentIndex().row() == 3)
    check("viewer shows frame 3", w.graphics.frames.current() == 3)
    w.graphics.frames.step('last')
    check("viewer steps to the last frame", w.graphics.frames.current() == 5)

    lp = w.graphics.logplot
    lp.combo.setCurrentIndex(lp.combo.findData('volume_uL'))
    check("log plot draws the points", len(lp.pts.points()) == 6)

    # rebuild with fewer columns as a background job
    from core import frame_log as fl
    w.rebuild_log(fl.column_keys(['volume_uL', 'R0_mm'], analyses=['swelling']))
    check("job runs", w.job_running() and w.a_stop.isEnabled())
    import time
    t0 = time.time()
    while w.job_running() and time.time() - t0 < 120:
        app.processEvents()
        time.sleep(0.02)
    app.processEvents()
    check("rebuild finished", not w.job_running())
    check("table shows the new columns", w.table.model().columnCount() == 7,
          str(w.folder.log.columns))
    check("log plot offers only logged columns", lp.combo.count() == 2)

    lpath = w.folder.log_path
    lpath.write_text(lpath.read_text(encoding='utf-8') + '\n', encoding='utf-8')
    w.open_session(str(folder.path))
    app.processEvents()
    check("problem shown in the tree", w.statuses['log'][0] == 'warn')
    w.go_next_required()
    check("Next goes to Frame Log", w.tree.currentItem() is w._items['log'])

    bad = tmp / 'not a session'
    bad.mkdir()
    n_warn = len(warnings)
    check("non-session folder refused", w.open_session(str(bad)) is False
          and len(warnings) == n_warn + 1)
    check("previous session still open", w.folder is not None)

    w.close_current()
    app.processEvents()
    check("closing returns to Start", w.folder is None and w.graphics.tabText(0) == 'Start')
    w.close()
    check("layout saved on close", ini.value('state') is not None)


if __name__ == "__main__":
    main()

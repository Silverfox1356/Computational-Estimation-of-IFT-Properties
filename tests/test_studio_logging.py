"""
Drop Analysis Studio, step 3 (gui.shell): set up and log a session from an
image, a video and a (fake) camera, on Qt's offscreen platform.  Drives the
real pages: the crop and scale tools are used by "drawing" on the frame,
the baseline is found automatically, the reference frame is measured, and
logging runs as a background job that opens the new folder when it ends.

Plain script (project convention — no pytest):
    conda run -n ift python tests/test_studio_logging.py      (about 2 min)
"""
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2

from _synthetic_video import render_drop, write_swelling_video

_passed = 0
_failed = 0


def check(name, cond, detail=""):
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  ok   {name}" + (f"  ({detail})" if detail else ""))
    else:
        _failed += 1
        print(f"  FAIL {name}" + (f"  ({detail})" if detail else ""))


class FakeCamera:
    """Stands in for core.camera_handler.CameraHandler: a drop that grows."""

    def __init__(self, camera_type):
        self.t0 = None
        self.stopped = False

    def start(self):
        self.t0 = time.monotonic()

    def get_frame(self):
        if self.t0 is None or self.stopped:
            return False, None
        R0 = 66 + 2 * min(time.monotonic() - self.t0, 5.0)
        return True, render_drop(0.30, R0)[0]

    def stop(self):
        self.stopped = True


def main():
    tmp = Path(tempfile.mkdtemp(prefix='studio logging '))
    try:
        run(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{_passed} passed, {_failed} failed")
    sys.exit(1 if _failed else 0)


def run(tmp):
    from PyQt6.QtCore import QSettings
    from PyQt6.QtWidgets import QApplication, QMessageBox, QPushButton
    app = QApplication.instance() or QApplication(sys.argv)
    warnings = []
    QMessageBox.warning = staticmethod(lambda *a, **k: warnings.append(a[2] if len(a) > 2 else ''))
    QMessageBox.information = staticmethod(lambda *a, **k: None)
    QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes)

    from gui.shell import nodes
    from gui.shell.main_window import StudioWindow
    ini = QSettings(str(tmp / 'studio.ini'), QSettings.Format.IniFormat)
    w = StudioWindow(settings_store=ini, camera_factory=FakeCamera)
    w.resize(1280, 800)
    w.show()

    def pump(seconds=0.05):
        t = time.time()
        while time.time() - t < seconds:
            app.processEvents()
            time.sleep(0.01)

    def wait_job(timeout=240):
        t = time.time()
        while w.job_running() and time.time() - t < timeout:
            app.processEvents()
            time.sleep(0.02)
        pump(0.1)
        return not w.job_running()

    def button(text):
        for b in w.settings.widget().findChildren(QPushButton):
            if b.text().startswith(text):
                return b
        raise AssertionError(f"no button '{text}' on the {w.settings.key} page")

    def drag(x0, y0, x1, y1):
        v = w.graphics.frames.view
        v.press_at(x0, y0)
        v.move_to(x1, y1)
        v.release_at(x1, y1)
        pump()

    # ---------------------------------------------------------------- image
    print("── image ──")
    img, _, _ = render_drop(0.30, 70)               # needle 80 px wide, tip at row 60
    png = tmp / 'drop.png'
    cv2.imwrite(str(png), img)
    check("new session from an image", w.new_session('image', path=str(png)))
    check("tree shows what is needed",
          w.statuses['scale'][0] == 'input' and w.statuses['baseline'][0] == 'input')
    check("start logging disabled until ready", not w.a_start.isEnabled())
    errors = []
    for n in nodes.NODES:
        try:
            w.select_node(n.key)
            pump(0.01)
        except Exception as e:
            errors.append(f"{n.key}: {e}")
    check("every draft page builds", not errors, '; '.join(errors))

    w.select_node('crop')
    button('Draw crop box').click()
    check("crop tool waits for a drag", w.graphics.frames.view.picking()
          and w.graphics.frames.hint.isVisible())
    drag(20, 0, 300, 350)
    check("crop applied from the drag", w.draft.crop_box() == (20, 0, 280, 350),
          str(w.draft.crop_box()))
    check("hint hidden again", not w.graphics.frames.hint.isVisible())

    w.select_node('scale')
    w.draft.needle['od_mm'] = 0.8
    w.select_node('scale')
    button('Draw box on needle').click()
    drag(90, 5, 190, 45)                             # needle is at x 100..180 after the crop
    s = w.draft.scale
    check("scale from the needle box", s is not None and abs(s['width_px'] - 80) <= 2,
          f"{(s or {}).get('width_px')} px")
    check("scale marker drawn", w.graphics.frames.scale_marker is not None)

    w.select_node('baseline')
    button('Find automatically').click()
    pump()
    check("baseline found", w.draft.baseline is not None, str(w.draft.baseline))
    check("tree complete for logging", w.statuses['logger'][0] == 'ready', w.statuses['logger'][1])
    check("start logging enabled", w.a_start.isEnabled())

    w.select_node('logger')
    button('Measure reference frame').click()
    check("reference fit finished", wait_job())
    check("reference measured", w.draft.preview is not None and w.draft.preview['fit_ok'])
    check("fit outline drawn", len(w.graphics.frames.view._overlays) >= 4,
          f"{len(w.graphics.frames.view._overlays)} overlay items")

    dest = tmp / 'image session'
    check("logging starts", w.start_logging(path=dest))
    check("logging finished", wait_job())
    check("session opened after logging", w.folder is not None and w.folder.path == dest
          and w.draft is None)
    check("one row logged", len(w.folder.log) == 1 and w.folder.log.rows[0]['fit_ok'])
    check("log sealed", w.folder.session.log_sha256 is not None and w.folder.problems == [])
    check("single image: analyses not available", w.statuses['swelling'][0] == 'off')

    # ---------------------------------------------------------------- video
    print("── video ──")
    vid = tmp / 'swell.avi'
    write_swelling_video(vid, lambda t: 1 + 0.05 * t / 4, duration=4, fps=10)
    check("new session from a video", w.new_session('video', path=str(vid)))
    d = w.draft
    d.set_scale_direct(10)
    d.auto_baseline()
    d.sampling.update(mode='interval', interval_s=1.0)
    w.draft_changed(rebuild=True)
    check("plan shown", d.sample_plan()['n'] == 5, f"{d.sample_plan()['n']} frames")
    dest = tmp / 'video session'
    progress_seen = []
    w.progress.bar.valueChanged.connect(progress_seen.append)
    check("video logging starts", w.start_logging(path=dest))
    check("video logging finished", wait_job())
    check("all planned frames logged", len(w.folder.log) == 5, f"{len(w.folder.log)} rows")
    check("progress reported", len(progress_seen) >= 3)
    v = w.folder.log.column('volume_uL')
    check("volume grows in the log", v[-1] > v[0], f"{v[0]:.4f} -> {v[-1]:.4f} uL")
    tabs = [w.graphics.tabText(i) for i in range(w.graphics.count())]
    check("frame and plot tabs", tabs == ['Frame', 'Log Plot'], str(tabs))
    check("table shows the rows", w.table.model().rowCount() == 5)

    # --------------------------------------------------------------- camera
    print("── camera ──")
    check("new session from the camera", w.new_session('camera', camera_type='Fake'))
    pump(0.3)
    check("live preview running", w.cam_timer.isActive() and w.latest_frame() is not None)
    w.select_node('source')
    pump(0.2)
    check("live picture shown on the Source page", w.graphics.frames.live_frame is not None)
    button('Capture reference frame').click()
    pump()
    d = w.draft
    d.set_scale_direct(10)
    d.auto_baseline()
    d.sampling.update(interval_s=0.2, duration_s=None)
    w.draft_changed(rebuild=True)
    dest = tmp / 'camera session'
    check("camera logging starts", w.start_logging(path=dest))
    t = time.time()
    while len(w.folder.log) < 2 and time.time() - t < 120:
        app.processEvents()
        time.sleep(0.02)
    w.cancel_job()
    check("camera logging stopped", wait_job())
    log = w.folder.log
    check("frames logged before Stop", len(log) >= 2, f"{len(log)} rows")
    check("clock time logged", log.has('wall_clock') and log.rows[0]['wall_clock'] != '')
    check("times increase", all(b['time_s'] > a['time_s'] for a, b in zip(log.rows, log.rows[1:])))
    check("camera released", w.camera is None and not w.cam_timer.isActive())
    check("stopped session sealed", w.folder.session.log_sha256 is not None
          and w.folder.problems == [])

    # ------------------------------------------------------------ discarding
    print("── discard ──")
    w.new_session('image', path=str(png))
    w.discard_draft()
    check("draft discarded", w.draft is None and w.graphics.tabText(0) == 'Start')
    check("no unexpected warnings", not warnings, '; '.join(map(str, warnings)))
    w.close()


if __name__ == "__main__":
    main()

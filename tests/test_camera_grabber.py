"""
Camera frames are grabbed on a background thread (core.camera_handler.FrameGrabber),
so a slow or stalled camera never freezes the Studio window.

    1  FrameGrabber keeps the newest frame, numbers it, survives grab errors, stops
    2  the Studio grabs off the window's thread; a camera that takes 0.5 s per
       frame leaves the window responsive; closing the session releases it

Plain script (project convention — no pytest):
    conda run -n ift python tests/test_camera_grabber.py      (a few seconds)
"""
import os
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _synthetic_video import render_drop

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


class SlowCamera:
    """A camera that takes ``delay`` s per frame and notes which thread asks."""

    delay = 0.5

    def __init__(self, camera_type=None):
        self.threads = set()
        self.stopped = False
        self.n = 0
        self.fail_at = set()
        self.frame = render_drop(0.30, 70)[0]

    def start(self):
        pass

    def get_frame(self):
        self.threads.add(threading.current_thread().name)
        self.n += 1
        if self.n in self.fail_at:
            raise RuntimeError('grab failed')
        if self.n > 1:                          # the first frame comes at once
            time.sleep(self.delay)
        return (not self.stopped), self.frame

    def stop(self):
        self.stopped = True


def main():
    print("── 1: FrameGrabber ──")
    from core.camera_handler import FrameGrabber
    cam = SlowCamera()
    cam.delay = 0.02
    cam.fail_at = {3}
    g = FrameGrabber(cam)
    g.start(first_frame=cam.frame)
    check("first frame available at once", g.latest()[0] >= 1 and g.latest()[1] is cam.frame)
    time.sleep(0.4)
    n = g.latest()[0]
    check("new frames counted", n > 5, f"{n} frames")
    check("a grab error is kept and grabbing goes on", g.error == 'RuntimeError: grab failed')
    check("grabs happen on the background thread", cam.threads == {'camera grabber'})
    check("stop ends the thread", g.stop(timeout_s=1.0))
    n = g.latest()[0]
    time.sleep(0.1)
    check("no grabs after stop", g.latest()[0] == n)

    print("── 2: the Studio ──")
    from PyQt6.QtCore import QSettings
    from PyQt6.QtWidgets import QApplication, QMessageBox
    app = QApplication.instance() or QApplication(sys.argv)
    QMessageBox.warning = staticmethod(lambda *a, **k: None)
    QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes)
    from gui.shell.main_window import StudioWindow
    tmp = Path(tempfile.mkdtemp(prefix='camera grabber '))
    cams = []

    def factory(camera_type):
        cams.append(SlowCamera(camera_type))
        return cams[-1]
    try:
        w = StudioWindow(settings_store=QSettings(str(tmp / 's.ini'),
                                                  QSettings.Format.IniFormat),
                         camera_factory=factory)
        w.resize(1280, 800)
        w.show()
        check("new session from a slow camera", w.new_session('camera', camera_type='Slow'))
        w.select_node('source')
        # The window should keep handling events while the camera takes 0.5 s a frame.
        worst, t_end = 0.0, time.monotonic() + 1.6
        while time.monotonic() < t_end:
            t = time.monotonic()
            app.processEvents()
            worst = max(worst, time.monotonic() - t)
            time.sleep(0.01)
        check("window stays responsive (no 0.5 s waits)", worst < 0.25,
              f"longest event pass {1000 * worst:.0f} ms")
        check("window thread never grabs after the first frame",
              cams[0].threads - {'MainThread'} == {'camera grabber'}
              and cams[0].n >= 3, f"{cams[0].n} grabs")
        check("live picture shown", w.graphics.frames.live_frame is not None)
        check("latest frame is a copy", w.latest_frame() is not cams[0].frame
              and w.latest_frame() is not None)
        t = time.monotonic()
        w.discard_draft()
        check("closing releases the camera", w.camera is None and w.grabber is None
              and cams[0].stopped and not w.cam_timer.isActive(),
              f"took {time.monotonic() - t:.2f} s")
        w.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{_passed} passed, {_failed} failed")
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()

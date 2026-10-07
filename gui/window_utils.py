"""
gui.window_utils
================

Behaviour every window and dialog in the app should have:

    * freely resizable – shrinking it below its content gives scroll bars
      instead of clipped or squashed widgets;
    * never opens larger than the screen it is on (laptops at 150 % scaling
      have only ~1280×800 logical pixels);
    * dialog buttons stay pinned at the bottom-right, outside the scrolling
      area, so they are always reachable.
"""

from PyQt6.QtWidgets import (QApplication, QFrame, QScrollArea, QVBoxLayout,
                             QWidget)

# Room left for the window title bar and a little breathing space.
_TITLE_BAR_PX = 32
_SCREEN_MARGIN_PX = 24


def scroll_root(window, layout_cls=QVBoxLayout):
    """Make ``window``'s whole content scrollable.

    Returns a new ``layout_cls`` to fill instead of ``layout_cls(window)``.
    The content stretches to fill the window when there is room and scrolls
    when the window is made smaller than the content's minimum size.
    """
    outer = QVBoxLayout(window)
    outer.setContentsMargins(0, 0, 0, 0)
    outer.setSpacing(0)
    inner = QWidget()
    outer.addWidget(_scroll_area_with(inner))
    return layout_cls(inner)


def scroll_body(outer_layout, layout_cls=QVBoxLayout):
    """Add a scrolling body to ``outer_layout`` and return its layout.

    For dialogs: put the fields in the returned layout, then add the button
    row to ``outer_layout`` so it stays pinned below the scrolling part.
    """
    inner = QWidget()
    outer_layout.addWidget(_scroll_area_with(inner), 1)
    lay = layout_cls(inner)
    lay.setContentsMargins(0, 0, 0, 0)
    return lay


def _scroll_area_with(inner):
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.Shape.NoFrame)
    scroll.setWidget(inner)
    return scroll


def fit_to_screen(window, width, height, center=True):
    """Resize to ``width``×``height`` but never beyond the available screen.

    ``center`` also places a top-level window in the middle of the screen
    (dialogs with a parent are positioned by Qt relative to their parent).
    """
    scr = window.screen() or QApplication.primaryScreen()
    avail = scr.availableGeometry()
    w = min(width, avail.width() - _SCREEN_MARGIN_PX)
    h = min(height, avail.height() - _SCREEN_MARGIN_PX - _TITLE_BAR_PX)
    window.resize(w, h)
    if center:
        window.move(avail.x() + (avail.width() - w) // 2,
                    avail.y() + max(0, (avail.height() - h - _TITLE_BAR_PX) // 2))
    return w, h

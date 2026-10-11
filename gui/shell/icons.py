"""
gui.shell.icons
===============

Line icons drawn from small SVG snippets (no emoji, no image files), plus
the node status icons.  ``icon(name)`` returns a cached QIcon rendered
crisply at the sizes the app uses.
"""

from PyQt6.QtCore import QByteArray, QRectF, Qt
from PyQt6.QtGui import QIcon, QPainter, QPixmap
from PyQt6.QtSvg import QSvgRenderer

_LINE = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
         'stroke="#35557f" stroke-width="1.4" stroke-linejoin="round" '
         'stroke-linecap="round">{}</svg>')

BODIES = {
    "new": '<path d="M6 3h8l4 4v14H6z" fill="#fff"/><path d="M14 3v4h4"/><path d="M12 11v6M9 14h6" stroke="#2f9e44" stroke-width="2"/>',
    "open": '<path d="M3 7h6l2 2h10v10H3z" fill="#f6d77a" stroke="#a7811a"/>',
    "close": '<path d="M3 7h6l2 2h10v10H3z" fill="#eee" stroke="#888"/><path d="M9 12l6 5M15 12l-6 5" stroke="#c0392b" stroke-width="1.6"/>',
    "save": '<path d="M4 4h13l3 3v13H4z" fill="#dce8f7"/><rect x="7" y="4" width="9" height="5" fill="#fff"/><rect x="7" y="13" width="10" height="7" fill="#fff"/>',
    "image": '<rect x="3" y="5" width="18" height="14" rx="1" fill="#eaf2fb"/><path d="M5 17l5-6 4 4 2-2 3 4z" fill="#7fb069" stroke="#4d7a3a"/><circle cx="16" cy="9" r="1.6" fill="#f2b632" stroke="none"/>',
    "video": '<rect x="3" y="6" width="13" height="12" rx="1.5" fill="#eaf2fb"/><path d="M16 10l5-3v10l-5-3z" fill="#cfe0f4"/>',
    "camera": '<path d="M3 8h4l2-2h6l2 2h4v11H3z" fill="#eaf2fb"/><circle cx="12" cy="13" r="3.5" fill="#cfe0f4"/>',
    "crop": '<path d="M7 3v14h14M3 7h14v14" stroke-width="2"/>',
    "threshold": '<circle cx="12" cy="12" r="8" fill="#fff"/><path d="M12 4a8 8 0 0 1 0 16z" fill="#35557f"/>',
    "ruler": '<rect x="2" y="8" width="20" height="8" rx="1" fill="#f7e9c6" stroke="#a4812b"/><path d="M6 8v3M10 8v4M14 8v3M18 8v4" stroke="#a4812b"/>',
    "baseline": '<path d="M12 4c3 4 5 7 5 9a5 5 0 0 1-10 0c0-2 2-5 5-9z" fill="#cfe0f4"/><path d="M2 20h20" stroke="#d64545" stroke-width="2"/>',
    "drop": '<path d="M9 2h6v3H9z" fill="#aaa" stroke="#666"/><path d="M12 5c-4 3-7 7-7 10a7 7 0 0 0 14 0c0-3-3-7-7-10z" fill="#4a7fc1" stroke="#2b5a94"/>',
    "series": '<path d="M3 3v18h18"/><circle cx="7" cy="9" r="1.3" fill="#2b6cb0"/><circle cx="11" cy="12" r="1.3" fill="#2b6cb0"/><circle cx="15" cy="14" r="1.3" fill="#2b6cb0"/><circle cx="19" cy="15" r="1.3" fill="#2b6cb0"/>',
    "swell": '<path d="M12 6c-3 2.5-5 5-5 7.5a5 5 0 0 0 10 0C17 11 15 8.5 12 6z" fill="#cfe0f4"/><path d="M3 13h2M19 13h2M12 21v1" stroke="#d9822b" stroke-width="2"/>',
    "mesh": '<path d="M3 20L12 4l9 16z" fill="#e3f0e0" stroke="#3d7a35"/><path d="M7.5 12h9M12 20l-4.5-8M12 20l4.5-8" stroke="#3d7a35"/>',
    "inv": '<circle cx="12" cy="12" r="8" fill="#fff"/><circle cx="12" cy="12" r="4"/><circle cx="12" cy="12" r="1" fill="#d64545" stroke="#d64545"/>',
    "plot": '<path d="M3 3v18h18"/><path d="M6 16l4-6 4 3 5-8" stroke="#d64545" stroke-width="1.8"/>',
    "report": '<path d="M6 3h9l4 4v14H6z" fill="#fff"/><path d="M9 11h7M9 14h7M9 17h5"/>',
    "table": '<rect x="3" y="4" width="18" height="16" fill="#fff"/><path d="M3 9h18M3 14h18M9 4v16"/>',
    "next": '<circle cx="12" cy="12" r="9" fill="#2b6cb0" stroke="#1d4f86"/><path d="M10 7l5 5-5 5" stroke="#fff" stroke-width="2.2"/>',
    "compute": '<rect x="3" y="3" width="18" height="18" rx="3" fill="#2f9e44" stroke="#22783a"/><path d="M7 10h10M7 14h10" stroke="#fff" stroke-width="2.2"/>',
    "stop": '<rect x="5" y="5" width="14" height="14" rx="2" fill="#d64545" stroke="#a82f2f"/>',
    "export": '<path d="M4 10v10h16V10"/><path d="M12 3v11M8 7l4-4 4 4" stroke="#2b6cb0" stroke-width="1.8"/>',
    "windows": '<rect x="3" y="4" width="18" height="16" fill="#fff"/><path d="M3 8h18M10 8v12"/>',
    "reset": '<path d="M5 12a7 7 0 1 0 2-5"/><path d="M5 3v4h4"/>',
    "rebuild": '<path d="M19 12a7 7 0 1 1-2-5"/><path d="M19 3v4h-4"/><path d="M9 12h6M12 9v6" stroke="#2f9e44" stroke-width="1.8"/>',
    "params": '<path d="M4 6h16M4 12h16M4 18h16"/><circle cx="9" cy="6" r="2" fill="#fff"/><circle cx="15" cy="12" r="2" fill="#fff"/><circle cx="7" cy="18" r="2" fill="#fff"/>',
    "calcurve": '<path d="M3 3v18h18"/><path d="M5 6c4 1 7 6 15 12" stroke="#7a4fbf" stroke-width="1.8"/>',
    "global": '<rect x="3" y="3" width="18" height="18" rx="3" fill="#7a4fbf" stroke="#5b3596"/><text x="12" y="17" font-size="13" font-family="Georgia" text-anchor="middle" fill="#fff" stroke="none">&#928;</text>',
    "project": '<path d="M12 3l8 4.5v9L12 21l-8-4.5v-9z" fill="#dce8f7"/><path d="M4 7.5l8 4.5 8-4.5M12 12v9"/>',
    "zoomfit": '<rect x="4" y="4" width="16" height="16"/><path d="M8 12h8M12 8v8"/>',
    "zin": '<circle cx="10" cy="10" r="6"/><path d="M15 15l5 5M7 10h6M10 7v6"/>',
    "zout": '<circle cx="10" cy="10" r="6"/><path d="M15 15l5 5M7 10h6"/>',
    "prev": '<path d="M6 5v14M18 5L9 12l9 7z" fill="#35557f"/>',
    "nextf": '<path d="M18 5v14M6 5l9 7-9 7z" fill="#35557f"/>',
    "first": '<path d="M5 5v14M12 5l-6 7 6 7M20 5l-6 7 6 7"/>',
    "last": '<path d="M19 5v14M12 5l6 7-6 7M4 5l6 7-6 7"/>',
    "help": '<circle cx="12" cy="12" r="9" fill="#fff"/><path d="M9.5 9.5a2.5 2.5 0 1 1 3.5 2.3c-.7.3-1 .8-1 1.5V14M12 17v.5" stroke-width="1.8"/>',
    "exit": '<path d="M10 4H4v16h6"/><path d="M14 8l4 4-4 4M8 12h10" stroke="#c0392b" stroke-width="1.8"/>',
}

_STATUS = {
    "done": '<circle cx="7" cy="7" r="6" fill="#2f9e44"/><path d="M4 7.2l2 2 4-4.4" stroke="#fff" stroke-width="1.7" fill="none"/>',
    "ready": '<circle cx="7" cy="7" r="5.3" fill="#fff" stroke="#2b6cb0" stroke-width="1.6"/>',
    "input": '<circle cx="7" cy="7" r="5.3" fill="#fff" stroke="#d64545" stroke-width="1.6"/><path d="M7 1.7a5.3 5.3 0 0 1 0 10.6z" fill="#d64545"/>',
    "warn": '<path d="M7 1.2l6.2 11.3H.8z" fill="#f0a020"/><path d="M7 5.3v3.6M7 10.4v.5" stroke="#fff" stroke-width="1.6"/>',
    "off": '<circle cx="7" cy="7" r="5.3" fill="#fff" stroke="#b5b5b5" stroke-width="1.4"/><path d="M4.5 7h5" stroke="#b5b5b5" stroke-width="1.4"/>',
    "stale": '<circle cx="7" cy="7" r="6" fill="#8a63d2"/><path d="M7 3.6V7l2.3 1.5" stroke="#fff" stroke-width="1.5" fill="none"/>',
}

_cache = {}


def _render(svg, sizes):
    renderer = QSvgRenderer(QByteArray(svg.encode('utf-8')))
    ic = QIcon()
    for s in sizes:
        pm = QPixmap(s, s)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        renderer.render(p, QRectF(0, 0, s, s))
        p.end()
        ic.addPixmap(pm)
    return ic


def icon(name):
    key = ('i', name)
    if key not in _cache:
        _cache[key] = _render(_LINE.format(BODIES[name]), (16, 20, 24, 32, 40, 48, 64))
    return _cache[key]


def status_icon(status):
    key = ('s', status)
    if key not in _cache:
        svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 14 14">'
               f'{_STATUS[status]}</svg>')
        _cache[key] = _render(svg, (13, 14, 16, 20, 26, 28))
    return _cache[key]

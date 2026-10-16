"""
gui.shell.guide
===============

The Guide: what each Analysis Builder node does and how its settings work.
The Settings pages keep only their controls and results; the explanations
they used to carry live here, one topic per node, and every page links to
its topic (the Guide link beside the breadcrumb, or Home > Guide).
"""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QDialog, QDialogButtonBox, QHBoxLayout, QListWidget,
                             QListWidgetItem, QSplitter, QTextBrowser, QVBoxLayout)

from gui.shell.nodes import NODE, NODES

# node key -> HTML.  Nodes without a topic open the Session topic.
TOPICS = {
    'session': """
<p>A session is one drop: its source (a video, an image or the camera), how the
frames were prepared, the frame log and the analyses run on it, all in one
folder.</p>
<p>A new session is set up on its reference frame (crop, threshold, scale,
baseline) and is saved when logging starts. The Analysis Builder shows what is
still needed; <b>Next</b> walks the steps in order.</p>""",
    'global': """
<p>Values that hold for the whole session: the fluids and the needle, and the
gamma(c) calibration curve.</p>""",
    'fluids': """
<p>The densities of the drop and of the surrounding phase and the needle's
outer diameter. Edit the values or pick a preset, then press <b>Apply</b>;
nothing changes until then.</p>
<ul>
<li>Presets are shared with the Pendant window's Settings. <b>Save</b> and
<b>Delete</b> work on your own presets only, not the built-in ones.</li>
<li>The inner diameter is used only by the diffusion model (Yang, forward
simulation), where the liquid inside the needle is part of the domain.
<i>Automatic</i> takes 70 % of the outer diameter.</li>
<li>The values are stored in the session when logging starts. After that only
the inner diameter can be changed; Yang and simulation results made with
another value become out of date.</li>
</ul>""",
    'calibration': """
<p>The interfacial tension against the dissolved-gas concentration, gamma(c),
measured beforehand. Only Diffusivity (Yang) and the forward simulation need
it. Loading validates the file and copies it into the session folder (for a
new session, when logging starts).</p>""",
    'source': """
<p>Where the frames come from.</p>
<ul>
<li><b>Video</b>: <i>Drop formed at</i> is the time the drop was formed.
Frames before it are not logged, and times in the log count from it. The frame
at this time is the reference frame you prepare on.</li>
<li><b>Camera</b>: the live picture shows on the Frame tab. Capture the
reference frame to prepare on; logging uses the same settings for every
frame.</li>
<li><b>Image</b>: one picture, logged as one frame at t = 0.</li>
</ul>""",
    'crop': """
<p>Crop to the needle and the drop: smaller frames are faster to fit and to
store. Changing the crop clears the baseline. A logged session stores its
frames already cropped.</p>""",
    'threshold': """
<p><i>Automatic</i> computes Otsu's threshold again on every frame;
<i>Fixed level</i> uses one grey level for all of them. Frames are stored
before thresholding, so the level can be changed later and the log rebuilt
(stored analysis results then become out of date). Tick <b>Show
thresholded</b> on the Frame tab to see what the fit sees.</p>""",
    'logger': """
<p>Which frames are logged and which columns the log keeps.</p>
<ul>
<li><b>Dense early, then sparser</b> spaces frames finely just after the drop
forms, when the IFT changes fastest. <b>One frame every N seconds</b> spaces
them evenly.</li>
<li>Camera: if a fit takes longer than the interval, the next frame is taken
as soon as it finishes; the log records the real times.</li>
<li>One fit gives every column, so leaving a column out saves almost no time.
A column left out can be added later by rebuilding the log. Columns the
analyses need are always logged.</li>
<li><b>Measure</b> fits the reference frame once, to check the set-up before
logging.</li>
</ul>
<p>Sampling is fixed once a session is logged: to change it, start a new
session from the same source.</p>""",
    'scale': """
<p>Converts pixels to metres.</p>
<ul>
<li><b>Needle width</b>: drag a tight box around a straight part of the
needle; its width is the needle's outer diameter.</li>
<li><b>Reference sphere</b>: drag a box around a sphere of known
diameter.</li>
<li><b>Known length</b>: click the two ends of a length you know.</li>
<li><b>Enter the scale</b>: type it in um/px.</li>
</ul>""",
    'baseline': """
<p>The row where the drop leaves the needle; only the outline below it is
fitted. <b>Automatic</b> looks for where the outline first widens beyond the
needle. In a logged session the row is in the coordinates of the stored
(cropped) frames.</p>""",
    'log': """
<p>One row per logged frame. <b>Rebuild</b> fits every stored frame again
(for example with other columns, or another threshold); the old log is
replaced only when all frames are done.</p>""",
    'frame_ift': """
<p>Fits one frame with the session's scale, baseline, fluids and threshold and
shows the full result, as the old Static mode did. Nothing is saved and the
frame log is not changed. <b>Image file</b> measures a picture taken with the
same camera, needle and magnification.</p>""",
    'analyses': """
<p>The analyses run on the logged frames: the swelling check first, then the
diffusivity method it recommends.</p>""",
    'swelling': """
<p>Compares the drop volume at the end of the log with the start. Above the
threshold the drop counts as swelling and Bagalkot's volume method is
recommended; below it, Yang's IFT method. (Bagalkot: ignoring 3 % swelling
shifts D by about 2 %.)</p>""",
    'yang': """
<p>Yang et al. (2006): the diffusion coefficient D and the mass-transfer
coefficient k from the drop's interfacial tension against time.</p>
<ul>
<li>Inputs: the drop outline (one frame becomes the model domain, mesh and
FEM), the gamma(c) calibration, and IFT against time for the same drop, from
the frame log or a CSV file.</li>
<li><b>Single frame</b>: with only one picture of the drop, its outline becomes
the model and IFT against time comes from a CSV. Run saves the session first
(it asks where).</li>
<li><b>kD grid</b> (Yang's Fig. 7a): for each kD (= k rn / D) the error E is
computed against D with forward simulations; the lowest point of all curves
gives D and kD, and k = kD D / rn. <b>Automatic</b> searches kD and D by
itself and widens the range until the minimum is inside it; <b>Manual
range</b> searches inside the ranges you give.</li>
<li><b>Time offset</b> is added to the curve's times if the drop formed before
its t = 0 (Yang et al. added 2 s).</li>
<li>The E-vs-D curves show on the Yang Fit tab.</li>
</ul>""",
    'yang_model': """
<p>The diffusion model is built from the drop outline of one frame whenever
Diffusivity (Yang) or the forward simulation runs: the domain (drop and needle
tip), a triangle mesh and the FEM system, with sanity checks (the Pendant
window's Steps 4 to 6 in one go). Details are on the Model tab.</p>""",
    'simulation': """
<p>The diffusion model of Yang et al. (2006) run forward for a chosen D and k:
how the gas enters the drop, the interface concentration and, with a gamma(c)
calibration, the IFT it gives. Diffusivity (Yang) runs it at its best fit by
itself.</p>
<ul>
<li><b>Automatic steps</b>: fine at first, growing later (as the Yang fit).</li>
<li><b>Fixed step</b>: uniform steps, theta = 0.7, as the Pendant window's
Step 7.</li>
</ul>""",
    'bagalkot': """
<p>Bagalkot's sphere model fitted to the drop volume against time: equally
spaced D values, then refined around the best. Gives D only (no k).
<b>Growing interface</b> lets the interface move outward as the drop swells
(Bagalkot thesis, section 4.2.4). The volume curve comes from the frame log or
a CSV file.</p>""",
    'results': """
<p>Every stored analysis result, with its warnings. Each result is stored in
the session folder (analyses/) with the settings it used and the log it was
computed from.</p>""",
    'report': """
<p>A printable report of the session and its analyses is planned.</p>""",
}


def topic(key):
    return key if key in TOPICS else 'session'


class GuideDialog(QDialog):
    """Topics on the left, the text on the right; resizable."""

    def __init__(self, key='session', parent=None):
        super().__init__(parent)
        self.setWindowTitle('Guide')
        self.resize(760, 500)
        self.list = QListWidget()
        self.list.setMaximumWidth(240)
        for n in NODES:
            if n.key in TOPICS:
                it = QListWidgetItem(n.label)
                it.setData(Qt.ItemDataRole.UserRole, n.key)
                self.list.addItem(it)
        self.text = QTextBrowser()
        self.text.setOpenLinks(False)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self.list)
        split.addWidget(self.text)
        split.setStretchFactor(1, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addWidget(split, 1)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(buttons)
        lay.addLayout(row)
        self.list.currentItemChanged.connect(self._show)
        self.show_topic(key)

    def show_topic(self, key):
        key = topic(key)
        for i in range(self.list.count()):
            if self.list.item(i).data(Qt.ItemDataRole.UserRole) == key:
                self.list.setCurrentRow(i)
                return

    def _show(self, item, _prev=None):
        if item is not None:
            key = item.data(Qt.ItemDataRole.UserRole)
            self.text.setHtml(f"<h3>{NODE[key].label}</h3>{TOPICS[key]}")

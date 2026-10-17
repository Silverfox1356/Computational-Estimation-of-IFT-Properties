# Drop Analysis Studio: design rules and building blocks

The Studio is a fixed frame (the "boilerplate") that every feature plugs into.
A new analysis, page or plot is built **only** from the blocks below, so it looks
and behaves like the rest of the app without new layout code. If a feature needs
something the blocks can't do, extend the block (and this file) for everyone,
rather than writing a one-off.

## 1. The frame of the window

| Area | What it is for | Code |
|---|---|---|
| Ribbon | Commands grouped by tab (Home, Analyses, ...) | `ribbon.py`, `main_window._make_ribbon` |
| Analysis Builder (tree) | One node per step; a status icon and a reason on every node | `nodes.py` (no Qt), `main_window` |
| Settings panel | The page of the selected node: settings, Run, result, warnings | `pages.py`, `draft_pages.py`, `analysis_pages.py` |
| Graphics area | Pictures and plots: one tab per kind (Frame, Live, result tabs) | `graphics.py`, `result_views.py` |
| Output panel | Messages, Progress, Table (frame log) | `panels.py` |

Side panels shrink, scroll and can be dragged; every window fits 1280 x 800,
is resizable, has scroll bars, buttons bottom-right, no emojis.

## 2. Rules

### Tree and status
- Every step is a `Node(key, parent, label, icon)` in `nodes.NODES`; the Next
  button follows `NEXT_ORDER`.
- Every node has a status from `STATUSES` (done, ready, input, warn, off, stale)
  **with a one-line reason**. A greyed-out or flagged node always says why.
  Statuses are computed in `nodes.py` (draft_statuses / node_statuses), never in
  the page code.

### Settings page (one per node)
Built top to bottom, always in this order:
1. Header: title, breadcrumb links, "Guide" link, status line. Built by
   `SettingsPanel.refresh`; never by the page itself.
2. **Inputs** section: what the step reads, `Can run: Yes / No: <reason>`.
3. **Parameters** section(s): editable values.
4. **Action buttons** via `actions(...)`: equal widths, right-aligned, primary
   action (Run, Apply, Start) first and blue (`_button(..., primary=True)`).
5. **Result** section: read-only `Section.field(label, value, unit, result=True)`.
6. **Warnings** section via `warnings_section(items)`: bullets, or "No warnings".
7. Secondary buttons (Show the plot, Export CSV...) in a `_row(...)`.

Text rules: button text is a short verb phrase ("Run", "Load...", "Apply");
"..." when a dialog follows; no icons on page buttons; no explanations on the
page. Explanations go in the Guide (`guide.TOPICS[node key]`) and tooltips.
Values wait for Apply when they change stored settings (Fluids & Needle).

### Warnings: one source, three places
- A result's warnings come from **one function**, `nodes.result_warnings`
  (no Qt, so it is testable without a display).
- They show in **the Warnings section of that node's page** (all of them),
  and the node's tree status turns "warn" with the first as the reason
  (`nodes.result_caveat`, "+n more on the page" when there are more).
  "Out of date" is its own tree status, not a warning there.
- Success notes are not warnings (e.g. Yang's "global minimum found").
- They never appear on a graphics tab; the line under the plots gives numbers only.
- Run-time events (started, finished, failed, stopped) go to the **Messages**
  panel: `messages.add(text, 'info' | 'ok' | 'warn' | 'error')`.
- Dialog boxes only for errors that stop the user (could not open, could not save).

### Graphics tabs
- A result or live tab is a `_View` subclass: plots are **tiles** on a grid of
  **equal cells**, declared in `LAYOUTS` as `(label, rows, cols, {tile: (row,
  col, rowspan, colspan)})`. The first layout is the default; the Layout box
  offers the rest and remembers the choice per tab; double-click enlarges a tile.
- Plots are made with `make_plot(title, xlabel, ylabel, logx, logy)`: white
  background, grid, legend top-right, `LogAxis` on log axes (labels never overlap).
- A plot-specific control (e.g. the kD picker) goes in its **tile header**
  (`tile.add_header(...)`), not in a separate row.
- Many series of one ordered quantity (kD): a sequential colour map plus a thin
  colour scale and a hover tooltip, not a long legend.
- A full-width text strip (a solver log) goes below the tiles: `add_strip`.

### Long work
- Anything slower than ~0.2 s runs as a job: `window.run_job(title, task,
  on_done, on_progress, on_finally)`; `task(progress, is_cancelled)` reports
  progress and checks for Stop. The window never waits on a fit, a solver or a
  camera (camera frames: `FrameGrabber`).
- Results are written to the session folder (`folder.save_analysis(name, data)`),
  so they show again when the session is reopened.

### Remembered choices
- Per-user choices (layouts, last folder, units) go in the window's
  `QSettings` store (`window.store`), keyed `area/name` (e.g. `layout/yang`).
- Anything that belongs to an experiment goes in the session folder, never in
  QSettings.

## 3. Building blocks

All page blocks are in **`gui/shell/kit.py`**; import them from there.

| Need | Use | Where |
|---|---|---|
| Collapsible section with a form | `Section(title)`, `.form.addRow`, `.field(...)`, `.widget(w)` | kit.py |
| Main buttons, bottom-right | `actions(*buttons)` | kit.py |
| A row of buttons that wraps | `button_row(*widgets)` / `flow_row(...)` | kit.py |
| A page button | `page_button(text, slot, primary=False, tip='')` | kit.py |
| Run button (disabled while a job runs) | `AnalysisPages._run_button(ok, slot)` | analysis_pages.py |
| Number box | `number_box(value, lo, hi, decimals, step, suffix)` | kit.py |
| Scientific number (1e-9) | `sci_edit(value)` / `parse_sci(text, name)` / `parse_sci_list` | kit.py |
| Frame rate (s/frame or fps) | `RateBox` | draft_pages.py |
| IFT result of one frame | `ift_result_section(rec)` | pages.py |
| Warnings list | `warnings_section(items)`; the items: `nodes.result_warnings` | kit.py / nodes.py |
| Highlighted note | `callout(text, 'warn' / 'good' / 'info')`, `note(text)` | kit.py |
| Number formatting | `fmt(value, digits)`; axis labels `sci_label(v)` | panels.py / result_views.py |
| Plot | `make_plot(...)`, `LogAxis` | result_views.py |
| Graphics tab | `_View` + `LAYOUTS` + `add_tile` + `finish_layout` | result_views.py |
| Background work | `run_job` / `Job` | main_window.py / panels.py |
| Messages | `window.messages.add(text, level)` | panels.py |
| Guide text | `guide.TOPICS[key]` | guide.py |
| Colours, sizes, styles | `theme.STYLESHEET` (objectNames: Primary, Note, Tile ...) | theme.py |

## 4. Recipe: adding a new analysis

1. **Core** (no Qt): `run_<name>(folder, ..., progress, is_cancelled)` in
   `core/analyses.py`, add to `RUNNERS` and `LABELS`, save with
   `folder.save_analysis`. Availability rule in `SessionFolder.availability`.
2. **Tree**: a `Node` in `nodes.NODES`, its place in `NEXT_ORDER`, its status
   rule in `node_statuses`.
3. **Page**: `AnalysisPages.page_<key>` in the order of section 2 (Inputs,
   Parameters, `actions(Run)`, Result, Warnings, secondary buttons).
4. **Warnings**: add its checks to `nodes.result_warnings` (page and tree follow).
5. **Graphics**: a `_View` subclass with `LAYOUTS`, tiles from `make_plot`;
   register it in `GraphicsArea.results` and `RESULT_TABS`.
6. **Guide**: a topic in `guide.TOPICS`.
7. **Tests**: checks in the matching `tests/test_studio_*.py` suite.

## 5. Known gaps

- The old Pendant / Sessile windows are outside these rules (kept as they are,
  by decision).

Tidied 2026-10-17: page blocks moved into `kit.py`; one warning source
(`nodes.result_warnings`) for page and tree; `LogPlot` built with `make_plot`.

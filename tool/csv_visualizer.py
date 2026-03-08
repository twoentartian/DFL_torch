#!/usr/bin/env python3
"""
CSV Visualizer — Interactive GUI for plotting CSV data.
  • Robust column-type handling (numeric coercion, datetime, categorical)
  • Multiple independent curve groups, each with own X / Y / Color config
  • Ctrl+scroll wheel to zoom in/out on the plot
  • Zoom, pan, and PDF export
  • Compact collapsible side panel — plot takes most of the window

Requirements:  pip install pandas matplotlib PyQt5
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.backends.backend_pdf import PdfPages

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QListWidget, QAbstractItemView,
    QGroupBox, QSplitter, QFrame, QMessageBox, QComboBox,
    QSizePolicy, QStatusBar, QScrollArea, QCheckBox, QToolButton
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QColor, QPalette, QWheelEvent


# ── Palette ──────────────────────────────────────────────────────────────────
BG       = "#0f1117"
PANEL    = "#1a1d27"
BORDER   = "#2a2d3e"
ACCENT   = "#4f8ef7"
ACCENT2  = "#a78bfa"
TEXT     = "#e2e8f0"
DIM      = "#64748b"
SUCCESS  = "#34d399"
WARNING  = "#fbbf24"
DANGER   = "#f87171"
PALETTE  = [ACCENT, ACCENT2, SUCCESS, WARNING, DANGER,
            "#38bdf8", "#fb923c", "#e879f9", "#a3e635", "#f472b6"]

STYLE = f"""
QMainWindow, QWidget {{
    background-color:{BG}; color:{TEXT};
    font-family:'Consolas','Courier New',monospace; font-size:11px;
}}
QGroupBox {{
    border:1px solid {BORDER}; border-radius:5px;
    margin-top:8px; padding-top:6px;
    font-weight:bold; color:{ACCENT}; font-size:10px; letter-spacing:1px;
}}
QGroupBox::title {{ subcontrol-origin:margin; left:8px; padding:0 4px; }}
QListWidget {{
    background-color:{PANEL}; border:1px solid {BORDER}; border-radius:4px;
    color:{TEXT}; selection-background-color:{ACCENT}; selection-color:white;
    outline:none; padding:1px;
}}
QListWidget::item {{ padding:3px 6px; border-radius:2px; }}
QListWidget::item:hover {{ background-color:#252840; }}
QListWidget::item:selected {{ background-color:{ACCENT}; color:white; }}
QPushButton {{
    background-color:{PANEL}; border:1px solid {BORDER}; border-radius:4px;
    color:{TEXT}; padding:4px 10px; font-family:'Consolas',monospace; font-size:11px;
}}
QPushButton:hover {{ border-color:{ACCENT}; color:{ACCENT}; }}
QPushButton:pressed {{ background-color:{ACCENT}; color:white; }}
QPushButton#primary {{ background-color:{ACCENT}; border-color:{ACCENT}; color:white; font-weight:bold; }}
QPushButton#primary:hover {{ background-color:#6fa0ff; }}
QPushButton#success {{ background-color:{SUCCESS}; border-color:{SUCCESS}; color:#0f1117; font-weight:bold; }}
QPushButton#danger  {{ background-color:{DANGER};  border-color:{DANGER};  color:white; font-weight:bold; }}
QPushButton#add     {{ background-color:#1e2535; border:1px dashed {ACCENT}; color:{ACCENT}; font-weight:bold; font-size:10px; }}
QPushButton#add:hover {{ background-color:{ACCENT}; color:white; }}
QComboBox {{
    background-color:{PANEL}; border:1px solid {BORDER}; border-radius:3px;
    color:{TEXT}; padding:3px 6px; font-size:11px;
}}
QComboBox::drop-down {{ border:none; width:16px; }}
QComboBox QAbstractItemView {{
    background-color:{PANEL}; border:1px solid {BORDER}; color:{TEXT};
    selection-background-color:{ACCENT};
}}
QLabel {{ color:{TEXT}; }}
QLabel#dim {{ color:{DIM}; font-size:10px; }}
QLabel#hdr {{ color:{ACCENT}; font-size:15px; font-weight:bold; letter-spacing:2px; }}
QLabel#curve_title {{ color:{TEXT}; font-weight:bold; font-size:11px; }}
QToolButton#collapse {{
    background:{PANEL}; border:1px solid {BORDER}; border-radius:3px;
    color:{DIM}; font-size:13px; padding:2px 5px;
}}
QToolButton#collapse:hover {{ border-color:{ACCENT}; color:{ACCENT}; }}
QSplitter::handle {{ background-color:{BORDER}; width:3px; }}
QStatusBar {{ background-color:{PANEL}; color:{DIM}; border-top:1px solid {BORDER}; font-size:10px; }}
QScrollBar:vertical {{ background:{BG}; width:6px; border-radius:3px; }}
QScrollBar::handle:vertical {{ background:{BORDER}; border-radius:3px; min-height:16px; }}
QScrollBar::handle:vertical:hover {{ background:{ACCENT}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height:0px; }}
QCheckBox {{ color:{TEXT}; spacing:4px; font-size:10px; }}
QCheckBox::indicator {{ width:12px; height:12px; border:1px solid {BORDER}; border-radius:2px; background:{PANEL}; }}
QCheckBox::indicator:checked {{ background:{ACCENT}; border-color:{ACCENT}; }}
"""


# ── Helper: coerce Series to plottable float64 ────────────────────────────────

def coerce_to_numeric(series: pd.Series):
    name = series.name
    warnings = []

    if pd.api.types.is_numeric_dtype(series):
        n_nan = int(series.isna().sum())
        if n_nan:
            warnings.append(f"'{name}': {n_nan} NaN value(s) will be skipped.")
        return series.astype(float), None, "numeric", " ".join(warnings)

    try:
        dt = pd.to_datetime(series, infer_datetime_format=True)
        numeric = (dt - dt.min()).dt.total_seconds()
        n_nan = int(numeric.isna().sum())
        if n_nan:
            warnings.append(f"'{name}': {n_nan} datetime parse failure(s) -> NaN.")
        return numeric, None, "datetime", " ".join(warnings)
    except Exception:
        pass

    cleaned = series.astype(str).str.replace(r"[^\d.\-eE+]", "", regex=True)
    coerced = pd.to_numeric(cleaned, errors='coerce')
    n_ok  = int(coerced.notna().sum())
    n_bad = int(coerced.isna().sum())
    if n_ok > 0 and n_bad / max(len(series), 1) < 0.5:
        if n_bad:
            warnings.append(f"'{name}': {n_bad} value(s) could not be parsed -> NaN.")
        return coerced.astype(float), None, "numeric", " ".join(warnings)

    cat    = series.astype("category")
    codes  = cat.cat.codes.astype(float)
    codes[codes == -1] = np.nan
    labels = list(cat.cat.categories.astype(str))
    warnings.append(
        f"'{name}' is non-numeric ({len(labels)} unique values) -> encoded as integers.")
    return codes, labels, "categorical", " ".join(warnings)


# ── Curve-group widget ────────────────────────────────────────────────────────

class CurveGroup(QWidget):
    def __init__(self, index: int, columns: list, parent=None):
        super().__init__(parent)
        self.index = index
        self._build(columns)

    def _build(self, columns):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(3)

        # Title row
        title_row = QHBoxLayout(); title_row.setSpacing(4)
        dot = QLabel("●")
        dot.setStyleSheet(f"color:{PALETTE[self.index % len(PALETTE)]}; font-size:13px;")
        title_row.addWidget(dot)
        lbl = QLabel(f"Group {self.index + 1}"); lbl.setObjectName("curve_title")
        title_row.addWidget(lbl)
        title_row.addStretch()
        self.enabled_cb = QCheckBox("On")
        self.enabled_cb.setChecked(True)
        title_row.addWidget(self.enabled_cb)
        outer.addLayout(title_row)

        # X column
        x_row = QHBoxLayout(); x_row.setSpacing(4)
        xl = QLabel("X:"); xl.setFixedWidth(18)
        x_row.addWidget(xl)
        self.x_combo = QComboBox()
        self.x_combo.addItems(columns)
        self.x_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        x_row.addWidget(self.x_combo)
        outer.addLayout(x_row)

        # Y columns
        yl = QLabel("Y (multi-select):"); yl.setObjectName("dim")
        outer.addWidget(yl)
        self.y_list = QListWidget()
        self.y_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.y_list.addItems(columns)
        self.y_list.setMaximumHeight(80)
        outer.addWidget(self.y_list)

        # Color column
        c_row = QHBoxLayout(); c_row.setSpacing(4)
        cl = QLabel("C:"); cl.setFixedWidth(18)
        c_row.addWidget(cl)
        self.color_combo = QComboBox()
        self.color_combo.addItem("— none —")
        self.color_combo.addItems(columns)
        self.color_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        c_row.addWidget(self.color_combo)
        outer.addLayout(c_row)

        line = QFrame(); line.setFrameShape(QFrame.HLine)
        line.setStyleSheet(f"color:{BORDER};")
        outer.addWidget(line)

    def get_config(self):
        if not self.enabled_cb.isChecked():
            return None
        return {
            "x":     self.x_combo.currentText(),
            "y":     [i.text() for i in self.y_list.selectedItems()],
            "color": None if self.color_combo.currentText() == "— none —"
                          else self.color_combo.currentText(),
        }


# ── Matplotlib canvas with Ctrl+scroll zoom ───────────────────────────────────

class MplCanvas(FigureCanvas):
    ZOOM_FACTOR = 1.15   # zoom step per wheel click

    def __init__(self, parent=None):
        self.fig = Figure(facecolor=BG)
        self.ax  = self.fig.add_subplot(111)
        self._style_ax(self.ax)
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setFocusPolicy(Qt.WheelFocus)

    def _style_ax(self, ax):
        ax.set_facecolor("#12151f")
        ax.tick_params(colors=DIM, labelsize=9)
        ax.xaxis.label.set_color(DIM)
        ax.yaxis.label.set_color(DIM)
        ax.title.set_color(TEXT)
        for sp in ax.spines.values():
            sp.set_color(BORDER)
        ax.grid(color=BORDER, linestyle='--', linewidth=0.5, alpha=0.6)

    def redraw(self):
        try:
            self.fig.tight_layout(pad=1.8)
        except Exception:
            pass
        self.draw()

    # ── Ctrl + scroll wheel zoom ──────────────────────────────────────────────
    def wheelEvent(self, event: QWheelEvent):
        modifiers = QApplication.keyboardModifiers()
        if modifiers != Qt.ControlModifier:
            # No Ctrl — let the default handler scroll the parent scroll area
            super().wheelEvent(event)
            return

        # Determine zoom direction
        delta = event.angleDelta().y()
        if delta == 0:
            return
        zoom_in = delta > 0

        # Get mouse position in data coordinates
        pos = event.pos()
        x_px, y_px = pos.x(), pos.y()
        # Convert widget pixel -> figure pixel -> data coords
        inv = self.ax.transData.inverted()
        # Map from Qt widget coords (y flipped) to matplotlib figure coords
        fig_h = self.fig.get_figheight() * self.fig.dpi
        x_data, y_data = inv.transform((x_px, fig_h - y_px))

        factor = 1.0 / self.ZOOM_FACTOR if zoom_in else self.ZOOM_FACTOR

        xl, xr = self.ax.get_xlim()
        yb, yt = self.ax.get_ylim()

        # Zoom centred on mouse pointer
        self.ax.set_xlim([
            x_data - (x_data - xl) * factor,
            x_data + (xr - x_data) * factor,
        ])
        self.ax.set_ylim([
            y_data - (y_data - yb) * factor,
            y_data + (yt - y_data) * factor,
        ])
        self.draw()


# ── Main window ───────────────────────────────────────────────────────────────

class CSVVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.df           = None
        self.filepath     = None
        self.curve_groups = []
        self._panel_visible = True
        self._build_ui()
        self.setWindowTitle("CSV Visualizer")
        self.resize(1340, 820)

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        self.setStyleSheet(STYLE)
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 4)
        root.setSpacing(6)

        # ── Compact header bar ──────────────────────────────────────────────
        hdr = QHBoxLayout(); hdr.setSpacing(8)

        # Toggle side panel button
        self.toggle_btn = QToolButton()
        self.toggle_btn.setObjectName("collapse")
        self.toggle_btn.setText("◀")
        self.toggle_btn.setToolTip("Hide/show controls panel")
        self.toggle_btn.setFixedSize(QSize(26, 26))
        self.toggle_btn.clicked.connect(self._toggle_panel)
        hdr.addWidget(self.toggle_btn)

        t = QLabel("CSV VISUALIZER"); t.setObjectName("hdr")
        hdr.addWidget(t)
        hdr.addStretch()

        self.load_btn = QPushButton("⊕ Load CSV")
        self.load_btn.setObjectName("primary")
        self.load_btn.setFixedHeight(26)
        self.load_btn.clicked.connect(self.load_csv)
        hdr.addWidget(self.load_btn)

        self.plot_btn = QPushButton("▶ Plot All")
        self.plot_btn.setObjectName("primary")
        self.plot_btn.setFixedHeight(26)
        self.plot_btn.clicked.connect(self.plot_all)
        hdr.addWidget(self.plot_btn)

        self.save_btn = QPushButton("⬇ Save PDF")
        self.save_btn.setObjectName("success")
        self.save_btn.setFixedHeight(26)
        self.save_btn.clicked.connect(self.save_pdf)
        hdr.addWidget(self.save_btn)

        self.file_lbl = QLabel("No file loaded"); self.file_lbl.setObjectName("dim")
        hdr.addWidget(self.file_lbl)

        # Zoom hint
        hint = QLabel("Ctrl+scroll to zoom")
        hint.setObjectName("dim")
        hint.setStyleSheet(f"color:{DIM}; font-size:10px; padding-left:8px;")
        hdr.addWidget(hint)

        # Wrap header in a fixed-height widget so it never expands
        hdr_widget = QWidget()
        hdr_widget.setFixedHeight(34)
        hdr_widget.setLayout(hdr)
        root.addWidget(hdr_widget)

        sep = QFrame(); sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"color:{BORDER};"); root.addWidget(sep)

        # ── Splitter: narrow left panel | wide plot area ────────────────────
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.setHandleWidth(3)
        root.addWidget(self.splitter, stretch=1)   # splitter takes all remaining space

        # ── Left panel (compact) ────────────────────────────────────────────
        self.left_widget = QWidget()
        self.left_widget.setMinimumWidth(210)
        self.left_widget.setMaximumWidth(240)
        lv = QVBoxLayout(self.left_widget)
        lv.setContentsMargins(0, 0, 4, 0)
        lv.setSpacing(4)

        cg_group = QGroupBox("Curve Groups")
        cg_v = QVBoxLayout(cg_group)
        cg_v.setSpacing(3); cg_v.setContentsMargins(3, 6, 3, 3)

        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"QScrollArea{{border:none; background:{BG};}}")
        self._cg_container = QWidget()
        self._cg_layout = QVBoxLayout(self._cg_container)
        self._cg_layout.setContentsMargins(0, 0, 0, 0); self._cg_layout.setSpacing(0)
        self._cg_layout.addStretch()
        scroll.setWidget(self._cg_container)
        cg_v.addWidget(scroll)

        btn_row = QHBoxLayout(); btn_row.setSpacing(3)
        add_btn = QPushButton("＋ Add")
        add_btn.setObjectName("add"); add_btn.setFixedHeight(24)
        add_btn.clicked.connect(self._add_curve_group)
        btn_row.addWidget(add_btn)

        rm_btn = QPushButton("✕ Remove")
        rm_btn.setObjectName("danger"); rm_btn.setFixedHeight(24)
        rm_btn.clicked.connect(self._remove_last_group)
        btn_row.addWidget(rm_btn)
        cg_v.addLayout(btn_row)

        lv.addWidget(cg_group, stretch=1)
        self.splitter.addWidget(self.left_widget)

        # ── Right: canvas ───────────────────────────────────────────────────
        right = QWidget()
        rv = QVBoxLayout(right); rv.setContentsMargins(0, 0, 0, 0); rv.setSpacing(0)

        self.canvas = MplCanvas(self)

        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setStyleSheet(f"""
            QToolBar{{ background:{PANEL}; border-bottom:1px solid {BORDER}; spacing:3px; padding:1px; }}
            QToolButton{{ background:transparent; color:{TEXT}; border-radius:3px; padding:2px; }}
            QToolButton:hover{{ background:{BORDER}; }}
        """)
        rv.addWidget(self.toolbar)
        rv.addWidget(self.canvas)
        self.splitter.addWidget(right)

        # Give the plot area ~85 % of the window width
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([220, 1100])

        self.status = QStatusBar(); self.setStatusBar(self.status)
        self.status.showMessage("Load a CSV file to begin.  |  Ctrl+scroll = zoom  |  ◀ button = hide panel")
        self._draw_placeholder()

    # ── Toggle side panel ─────────────────────────────────────────────────────

    def _toggle_panel(self):
        self._panel_visible = not self._panel_visible
        self.left_widget.setVisible(self._panel_visible)
        self.toggle_btn.setText("◀" if self._panel_visible else "▶")

    # ── Placeholder ───────────────────────────────────────────────────────────

    def _draw_placeholder(self):
        ax = self.canvas.ax; ax.clear(); self.canvas._style_ax(ax)
        ax.text(0.5, 0.5, "Load a CSV  →  ▶ Plot All",
                transform=ax.transAxes, ha='center', va='center',
                fontsize=13, color=DIM, fontfamily='monospace')
        ax.set_xticks([]); ax.set_yticks([])
        self.canvas.draw()

    # ── Curve-group management ────────────────────────────────────────────────

    def _add_curve_group(self, columns=None):
        cols = columns or (list(self.df.columns) if self.df is not None else [])
        if not cols:
            QMessageBox.warning(self, "No data", "Please load a CSV file first.")
            return
        idx = len(self.curve_groups)
        grp = CurveGroup(idx, cols)
        self.curve_groups.append(grp)
        self._cg_layout.insertWidget(self._cg_layout.count() - 1, grp)

    def _remove_last_group(self):
        if not self.curve_groups:
            return
        grp = self.curve_groups.pop()
        self._cg_layout.removeWidget(grp)
        grp.deleteLater()

    # ── Load CSV ──────────────────────────────────────────────────────────────

    def load_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open CSV", "", "CSV Files (*.csv);;TSV Files (*.tsv);;All Files (*)")
        if not path:
            return
        try:
            sep = "\t" if path.lower().endswith(".tsv") else ","
            self.df = pd.read_csv(path, sep=sep)
            self.filepath = path
            self.file_lbl.setText(os.path.basename(path))
            while self.curve_groups:
                self._remove_last_group()
            self._add_curve_group(list(self.df.columns))
            nr, nc = self.df.shape
            self.status.showMessage(
                f"Loaded: {os.path.basename(path)}  —  {nr} rows × {nc} cols")
        except Exception as e:
            QMessageBox.critical(self, "Error loading CSV", str(e))

    # ── Plot ──────────────────────────────────────────────────────────────────

    def plot_all(self):
        if self.df is None:
            QMessageBox.warning(self, "No data", "Please load a CSV file first.")
            return

        ax = self.canvas.ax
        ax.clear(); self.canvas._style_ax(ax)

        all_warnings = []
        plotted      = 0
        color_idx    = 0

        for grp in self.curve_groups:
            cfg = grp.get_config()
            if cfg is None:
                continue

            x_col   = cfg["x"]
            y_cols  = cfg["y"]
            col_col = cfg["color"]

            if not y_cols:
                all_warnings.append(f"Group {grp.index+1}: no Y columns selected — skipped.")
                continue

            x_num, x_labels, x_dtype, x_warn = coerce_to_numeric(self.df[x_col])
            if x_warn:
                all_warnings.append(f"[X '{x_col}'] {x_warn}")

            use_cmap = False
            if col_col and col_col in self.df.columns:
                c_num, _, c_dtype, c_warn = coerce_to_numeric(self.df[col_col])
                if c_warn:
                    all_warnings.append(f"[Color '{col_col}'] {c_warn}")
                c_vmin, c_vmax = np.nanmin(c_num), np.nanmax(c_num)
                if c_vmin == c_vmax:
                    c_vmax = c_vmin + 1
                c_norm = plt.Normalize(c_vmin, c_vmax)
                c_cmap = cm.get_cmap("plasma")
                use_cmap = True

            for y_col in y_cols:
                if y_col not in self.df.columns:
                    all_warnings.append(f"Column '{y_col}' not found — skipped.")
                    continue

                y_num, y_labels, y_dtype, y_warn = coerce_to_numeric(self.df[y_col])
                if y_warn:
                    all_warnings.append(f"[Y '{y_col}'] {y_warn}")

                mask = x_num.notna() & y_num.notna()
                n_dropped = int((~mask).sum())
                if n_dropped:
                    all_warnings.append(
                        f"'{x_col}' vs '{y_col}': {n_dropped} NaN row(s) dropped.")

                x_plot = x_num[mask].values
                y_plot = y_num[mask].values

                if len(x_plot) == 0:
                    all_warnings.append(f"'{y_col}': no valid data remains.")
                    continue

                base_color = PALETTE[color_idx % len(PALETTE)]

                if use_cmap:
                    c_vals = c_num[mask].values
                    for j in range(len(x_plot) - 1):
                        cv = c_vals[j]
                        seg_c = (c_cmap(c_norm(cv)) if not np.isnan(cv)
                                 else (0.5, 0.5, 0.5, 1.0))
                        ax.plot(x_plot[j:j+2], y_plot[j:j+2],
                                color=seg_c, linewidth=1.5, solid_capstyle='round')
                    ax.plot([], [], color=base_color, linewidth=1.5,
                            label=f"G{grp.index+1}: {y_col} [col={col_col}]")
                else:
                    ax.plot(x_plot, y_plot, color=base_color, linewidth=1.8,
                            label=f"G{grp.index+1}: {y_col}", solid_capstyle='round')

                if x_dtype == "categorical" and x_labels:
                    unique_codes = np.unique(x_plot.astype(int))
                    ax.set_xticks(unique_codes)
                    ax.set_xticklabels(
                        [x_labels[c] if c < len(x_labels) else str(c) for c in unique_codes],
                        rotation=45, ha='right', fontsize=8, color=DIM)
                elif x_dtype == "datetime":
                    try:
                        x0 = pd.to_datetime(self.df[x_col], infer_datetime_format=True).min()
                        ticks = ax.get_xticks()
                        ax.set_xticklabels(
                            [(x0 + pd.Timedelta(seconds=float(t))).strftime("%Y-%m-%d %H:%M")
                             for t in ticks if not np.isnan(float(t))],
                            rotation=30, ha='right', fontsize=8, color=DIM)
                    except Exception:
                        pass

                if y_dtype == "categorical" and y_labels:
                    unique_codes = np.unique(y_plot.astype(int))
                    ax.set_yticks(unique_codes)
                    ax.set_yticklabels(
                        [y_labels[c] if c < len(y_labels) else str(c) for c in unique_codes],
                        fontsize=8, color=DIM)

                color_idx += 1
                plotted   += 1

            if use_cmap:
                sm = plt.cm.ScalarMappable(cmap=c_cmap, norm=c_norm)
                sm.set_array([])
                cbar = self.canvas.fig.colorbar(sm, ax=ax, pad=0.01, fraction=0.025)
                cbar.set_label(col_col, color=DIM, fontsize=9)
                cbar.ax.yaxis.set_tick_params(color=DIM, labelcolor=DIM)
                cbar.outline.set_edgecolor(BORDER)

        if plotted == 0:
            self._draw_placeholder()
            if all_warnings:
                QMessageBox.warning(self, "Nothing plotted",
                    "No valid data found.\n\n" + "\n".join(all_warnings))
            return

        ax.legend(facecolor=PANEL, edgecolor=BORDER,
                  labelcolor=TEXT, fontsize=9, framealpha=0.9)
        all_x = list({grp.get_config()["x"]
                      for grp in self.curve_groups if grp.get_config()})
        ax.set_xlabel(", ".join(all_x), color=DIM, fontsize=10)

        self.canvas.redraw()

        msg = f"Plotted {plotted} curve(s).  Ctrl+scroll = zoom."
        if all_warnings:
            msg += f"  ({len(all_warnings)} warning(s))"
            self.status.showMessage(msg)
            dlg = QMessageBox(self)
            dlg.setWindowTitle("Plot Warnings")
            dlg.setIcon(QMessageBox.Warning)
            dlg.setText(f"{len(all_warnings)} issue(s) while plotting:")
            dlg.setDetailedText("\n".join(all_warnings))
            dlg.exec_()
        else:
            self.status.showMessage(msg)

    # ── Save PDF ──────────────────────────────────────────────────────────────

    def save_pdf(self):
        if self.df is None:
            QMessageBox.warning(self, "Nothing to save", "Plot something first.")
            return
        default = "plot.pdf"
        if self.filepath:
            base    = os.path.splitext(os.path.basename(self.filepath))[0]
            default = f"{base}_plot.pdf"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save PDF", default, "PDF Files (*.pdf)")
        if not path:
            return
        try:
            with PdfPages(path) as pdf:
                orig = self.canvas.fig.get_dpi()
                self.canvas.fig.set_dpi(150)
                pdf.savefig(self.canvas.fig, bbox_inches='tight',
                            facecolor=self.canvas.fig.get_facecolor())
                self.canvas.fig.set_dpi(orig)
            self.status.showMessage(f"PDF saved -> {path}")
            QMessageBox.information(self, "Saved", f"PDF saved to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Save failed", str(e))


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    pal = QPalette()
    for role, hex_ in [
        (QPalette.Window,          BG),
        (QPalette.WindowText,      TEXT),
        (QPalette.Base,            PANEL),
        (QPalette.AlternateBase,   BG),
        (QPalette.Text,            TEXT),
        (QPalette.Button,          PANEL),
        (QPalette.ButtonText,      TEXT),
        (QPalette.Highlight,       ACCENT),
        (QPalette.HighlightedText, "#ffffff"),
    ]:
        pal.setColor(role, QColor(hex_))
    app.setPalette(pal)

    win = CSVVisualizer()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

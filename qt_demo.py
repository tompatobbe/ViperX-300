#!/usr/bin/env python3
"""PyQt5 capability tour — tabs, live plot, table, threads. Run: python3 qt_demo.py"""
import sys, math, random
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QPainter, QPen, QColor
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QLineEdit, QCheckBox, QComboBox, QProgressBar,
    QTableWidget, QTableWidgetItem, QFormLayout, QTextEdit, QFileDialog, QMessageBox,
)


class WidgetsTab(QWidget):
    """Common input widgets wired to a live status label."""
    def __init__(self):
        super().__init__()
        form = QFormLayout(self)
        self.status = QLabel("ready")

        line = QLineEdit("type here")
        line.textChanged.connect(lambda t: self.status.setText(f"text: {t}"))

        slider = QSlider(Qt.Horizontal)
        slider.valueChanged.connect(lambda v: self.status.setText(f"slider: {v}"))

        combo = QComboBox()
        combo.addItems(["waist", "shoulder", "elbow", "wrist"])
        combo.currentTextChanged.connect(lambda t: self.status.setText(f"joint: {t}"))

        check = QCheckBox("enable")
        check.toggled.connect(lambda b: self.status.setText(f"enabled: {b}"))

        pick = QPushButton("Open file…")
        pick.clicked.connect(self._pick)

        form.addRow("Line edit", line)
        form.addRow("Slider", slider)
        form.addRow("Combo", combo)
        form.addRow("Checkbox", check)
        form.addRow("Dialog", pick)
        form.addRow("Status", self.status)

    def _pick(self):
        path, _ = QFileDialog.getOpenFileName(self, "Pick any file")
        if path:
            self.status.setText(f"chose: {path}")


class PlotWidget(QWidget):
    """Hand-drawn scrolling sine plot via QPainter + QTimer — no plotting dep."""
    def __init__(self):
        super().__init__()
        self.phase = 0.0
        self.setMinimumHeight(200)
        t = QTimer(self)
        t.timeout.connect(self._tick)
        t.start(30)

    def _tick(self):
        self.phase += 0.1
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(self.rect(), QColor(20, 20, 30))
        p.setPen(QPen(QColor(80, 80, 100), 1))
        p.drawLine(0, h // 2, w, h // 2)
        p.setPen(QPen(QColor(0, 200, 160), 2))
        prev = None
        for x in range(w):
            y = h / 2 + (h / 3) * math.sin(0.03 * x + self.phase)
            if prev:
                p.drawLine(prev[0], int(prev[1]), x, int(y))
            prev = (x, y)


class Worker(QThread):
    """Background work that streams progress without freezing the UI."""
    progress = pyqtSignal(int)
    done = pyqtSignal(str)

    def run(self):
        total = 0
        for i in range(101):
            total += random.random()
            self.msleep(20)
            self.progress.emit(i)
        self.done.emit(f"accumulated {total:.2f}")


class ThreadTab(QWidget):
    def __init__(self):
        super().__init__()
        v = QVBoxLayout(self)
        self.bar = QProgressBar()
        self.log = QTextEdit(readOnly=True)
        btn = QPushButton("Run background job")
        btn.clicked.connect(self._start)
        v.addWidget(btn)
        v.addWidget(self.bar)
        v.addWidget(self.log)
        self.worker = None

    def _start(self):
        if self.worker and self.worker.isRunning():
            return
        self.worker = Worker()
        self.worker.progress.connect(self.bar.setValue)
        self.worker.done.connect(lambda m: self.log.append(f"done: {m}"))
        self.worker.start()
        self.log.append("started…")


class TableTab(QWidget):
    def __init__(self):
        super().__init__()
        v = QVBoxLayout(self)
        cols = ["joint", "pos", "vel", "effort"]
        joints = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
        t = QTableWidget(len(joints), len(cols))
        t.setHorizontalHeaderLabels(cols)
        for r, j in enumerate(joints):
            t.setItem(r, 0, QTableWidgetItem(j))
            for c in range(1, 4):
                t.setItem(r, c, QTableWidgetItem(f"{random.uniform(-1, 1):.3f}"))
        v.addWidget(QLabel("Editable table (double-click a cell):"))
        v.addWidget(t)


class Main(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt5 capability tour")
        self.resize(640, 480)
        tabs = QTabWidget()
        tabs.addTab(WidgetsTab(), "Widgets")
        plot = QWidget()
        pv = QVBoxLayout(plot)
        pv.addWidget(QLabel("Custom QPainter animation (60fps-ish):"))
        pv.addWidget(PlotWidget())
        tabs.addTab(plot, "Live plot")
        tabs.addTab(ThreadTab(), "Threads")
        tabs.addTab(TableTab(), "Table")
        self.setCentralWidget(tabs)

        bar = self.menuBar().addMenu("Help")
        bar.addAction("About", lambda: QMessageBox.information(self, "About", "PyQt5 demo"))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = Main()
    win.show()
    sys.exit(app.exec_())

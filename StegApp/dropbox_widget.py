from pathlib import Path
from PyQt5 import QtCore, QtGui, QtWidgets

class DropBox(QtWidgets.QGroupBox):
    fileDropped = QtCore.pyqtSignal(Path)

    def __init__(self, title: str):
        super().__init__(title)
        self.setObjectName("dropZone")
        self.setProperty("dragOver", False)
        self.setAcceptDrops(True)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(12, 30, 12, 14)
        lay.setSpacing(8)

        row = QtWidgets.QHBoxLayout()
        row.setAlignment(QtCore.Qt.AlignCenter)

        self.label = QtWidgets.QLabel("Drop a file here or")
        self.label.setAlignment(QtCore.Qt.AlignVCenter)

        self.browse_btn = QtWidgets.QPushButton("Browse")
        self.browse_btn.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.browse_btn.setObjectName("browseBtnInline")
        self.browse_btn.clicked.connect(self._open_file_dialog)

        row.addWidget(self.label)
        row.addWidget(self.browse_btn)

        lay.addStretch(1)
        lay.addLayout(row)
        lay.addStretch(1)

        self.setMinimumHeight(120)

    def _open_file_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select File")
        if path:
            p = Path(path)
            self.label.setText(f"Selected: {p.name}")
            self.fileDropped.emit(p)

    def dragEnterEvent(self, e: QtGui.QDragEnterEvent):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
            self.setProperty("dragOver", True)
            self.style().unpolish(self); self.style().polish(self)

    def dropEvent(self, e: QtGui.QDropEvent):
        urls = e.mimeData().urls()
        if not urls:
            self.setProperty("dragOver", False)
            self.style().unpolish(self); self.style().polish(self)
            return
        p = Path(urls[0].toLocalFile())
        self.label.setText(p.name)
        self.fileDropped.emit(p)
        self.setProperty("dragOver", False)
        self.style().unpolish(self); self.style().polish(self)

    def dragLeaveEvent(self, e: QtGui.QDragLeaveEvent):
        self.setProperty("dragOver", False)
        self.style().unpolish(self); self.style().polish(self)
        e.accept()

import sys
from PyQt5 import QtWidgets
from ui import MainWindow

def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow(); w.resize(1100, 720); w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

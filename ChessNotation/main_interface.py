import sys

from Interface.MainWindow import *

if __name__ == "__main__":
    app = QApplication(sys.argv)
    # print(PyQt6.QtWidgets.QStyleFactory.keys())
    # app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

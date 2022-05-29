from GUI.flatten_stack import Flatten_stack_window
from PyQt5 import QtCore, QtGui, QtWidgets
import sys
app = QtWidgets.QApplication(sys.argv)
window = Flatten_stack_window()
window.show()
sys.exit(app.exec())
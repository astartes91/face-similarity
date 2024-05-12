import sys

from PyQt6.QtWidgets import QApplication

from MyWindowClass import MyWindowClass

app = QApplication(sys.argv)
myWindow = MyWindowClass(None)
myWindow.show()
app.exec()
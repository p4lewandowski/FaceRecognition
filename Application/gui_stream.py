import sys
from PyQt5 import QtGui, QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, \
    QTextEdit, QDesktopWidget

class Stream(QtCore.QObject):
    newText = QtCore.pyqtSignal(str)

    def write(self, text):
        self.newText.emit(str(text))

    def flush(self):
        pass

class GUI_ConsoleOutput(QWidget):
    def __init__(self):
        super(GUI_ConsoleOutput, self).__init__()
        self.setGeometry(50, 50, 650, 300)
        self.setWindowTitle("PyQT tuts!")
        self.setWindowIcon(QtGui.QIcon('pythonlogo.png'))
        self.home()

        sys.stdout = Stream(newText=self.onUpdateText)

    def onUpdateText(self, text):
        cursor = self.process.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.process.setTextCursor(cursor)
        self.process.ensureCursorVisible()

    def __del__(self):
        sys.stdout = sys.__stdout__

    def home(self):

        lay = QVBoxLayout(self)

        self.process  = QTextEdit()
        self.process.moveCursor(QtGui.QTextCursor.Start)
        self.process.ensureCursorVisible()
        self.process.setLineWrapColumnOrWidth(500)
        self.process.setLineWrapMode(QTextEdit.FixedPixelWidth)
        btn = QPushButton("Close")
        btn.clicked.connect(self.dothings)

        lay.addWidget(self.process)
        lay.addWidget(btn)

        self.centre()
        self.show()

    def centre(self):
        """
        Center the window on screen. This implemention will handle the window
        being resized or the screen resolution changing.
        """
        screen = QDesktopWidget().screenGeometry()
        mysize = self.geometry()
        hpos = (screen.width() - mysize.width()) / 2
        vpos = (screen.height() - mysize.height()) / 2
        self.move(hpos, vpos)

    def dothings(self):
        print("yo mama")

def run():
    app = QApplication(sys.argv)
    GUI = GUI_ConsoleOutput()
    sys.exit(app.exec_())

# run()
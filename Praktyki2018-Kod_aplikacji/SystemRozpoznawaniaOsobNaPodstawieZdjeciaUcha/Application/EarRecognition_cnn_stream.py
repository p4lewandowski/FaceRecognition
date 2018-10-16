import sys
from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QTextEdit, QDesktopWidget
from keras import callbacks
from PyQt5.QtCore import QObject, pyqtSignal, QCoreApplication, Qt


class OutputStream(callbacks.Callback):
    """
    Stworzony niestandardowy objekt typu callback umożliwiający dodatkową kontrolę wyświetlania w
    trakcie procesu uczenia sieci neuronowej. W tym przypadku callback umożliwia wyświetlanie stanu
    procesu uczenia w czasie rzeczywistym w aplikacji GUI.
    """
    def on_train_begin(self, logs={}):
        """
        W trakcie rozpoczecia treningu zarezerwuj czas dla aplikacji, aby mogła ona wyświetlać
        proces uczenia w aplikacji w czasie rzeczywistym.
        """

        QCoreApplication.processEvents()

    def on_batch_end(self, batch, logs={}):
        """
        Po skończeniu każdego batcha zarezerwuj czas dla aplikacji, aby mogła ona wyświetlać
        proces uczenia w aplikacji czasie rzeczywistym.
        """
        QCoreApplication.processEvents()

class EmittingStream(QObject):

    textWritten = pyqtSignal(str)

    def write(self, text):
        self.textWritten.emit(str(text))

    def flush(self):
        pass

class GUI_ConsoleOutput(QWidget):
    """
    Klasa inicjalizuje QWidget stworzony w celu wyświetlania procesu uczenia sieci neuronowej w
    oddzielnym oknie. Niezbędnym jest utworzenie nowego strumienia wyjścia i użycie go zamiast
    domyślnego strumienia wyjścia, który wyświetla informacje w konsoli.
    """
    def __init__(self):
        super(GUI_ConsoleOutput, self).__init__()
        self.setGeometry(50, 50, 720, 300)
        self.setWindowTitle("Trening sieci w toku")
        self.setWindowIcon(QtGui.QIcon('pythonlogo.png'))
        screen = QDesktopWidget().screenGeometry()
        mysize = self.geometry()
        hpos = (screen.width() - mysize.width()) / 2
        vpos = (screen.height() - mysize.height()) / 2
        self.move(hpos, vpos)

        # Osadzenie elementów w układzie.
        lay = QVBoxLayout(self)
        self.btn = QPushButton("Zamknij")
        self.btn.clicked.connect(self.close_status)
        self.btn.setEnabled(False)
        self.textEdit = QTextEdit()
        self.textEdit.setReadOnly(True)
        self.textEdit.setTextInteractionFlags(Qt.NoTextInteraction)

        lay.addWidget(self.textEdit)
        lay.addWidget(self.btn)

        # Przekierowanie wyjścia do innego streamu
        sys.stdout = EmittingStream(textWritten=self.normalOutputWritten)

        self.show()

    def close_status(self):
        """
        Funkcja zamyka wywołane wcześniej okno do monitorowania procesu uczenia.
        """
        self.close()

    def normalOutputWritten(self, text):
        """
        Aplikacja umożliwia wyświetlanie informacji z wcześniej stworzonego strumienia wyjścia w
        oknie aplikacji.

        Args:
            text: Tekst do wyświetlenia w oknie.
        """
        cursor = self.textEdit.textCursor()
        cursor.insertText(text)
        # Zablokowanie kursora do czasu skończenia procesu uczenia
        self.textEdit.setTextCursor(cursor)
        self.textEdit.ensureCursorVisible()
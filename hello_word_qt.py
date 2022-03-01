import sys
import time

from PyQt5 import QtCore
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication, QMainWindow, QGridLayout, QWidget, QTableWidget, QTableWidgetItem
from PyQt5.QtCore import QSize, Qt


from text_bot_gui import Ui_MainWindow

def answer_the_question():
    return 'No'


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setupUi(self)

        self.font = QFont()
        self.horizontalSlider.setMinimum(0)
        self.horizontalSlider.setMaximum(30)

        self.label.setGeometry(QtCore.QRect(150,70,70,50))
        self.label.setText('Bot')
        self.horizontalSlider.valueChanged.connect(self.lcdNumber.display)

        self.pushButton.clicked.connect(self.show_answer)

    def value_(self,value):
        print(value)

    def show_answer(self):
        self.jc = self.input_text_line.toPlainText()
        print('HERE')
        self.input_text_line.clear()
        print('HERE')
        time.sleep(0.5)
        s = answer_the_question()
        print(s)
        self.Main_text_window.setText(s)
        print('HERE')




def main():
    app = QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())

if __name__ == '__main__':  # Если мы запускаем файл напрямую, а не импортируем
    main()  # то запускаем функцию main()

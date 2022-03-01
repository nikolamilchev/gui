import random
import sys
import time

from PyQt5 import QtCore
from PyQt5 import QtCore, QtGui, QtWidgets, Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication, QMainWindow, QGridLayout, QWidget, QTableWidget, QTableWidgetItem, \
    QSizePolicy, QPushButton, QVBoxLayout
from PyQt5.QtCore import QSize, Qt
from  matplotlib.pyplot import figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar


from text_bot_gui_with_mpl import Ui_MainWindow
import numpy as np
from matplotlib import pyplot as plt

class MyMplCanvas(FigureCanvas):
    def __init__(self):

        self.fig = plt.figure
        FigureCanvas.__init__(self,self.fig)
        FigureCanvas.setSizePolicy(self,QSizePolicy.Expanding,QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

        self.companovka_for_mpl = QtGui.QVBoxLayout(self.widget)
        self.canvas = MyMplCanvas(self.fig)
        self.companovka_for_mpl.addWidget(self.canvas )
        self.toolbar = NavigationToolbar(self.canvas,self)

def answer_the_question():
    return 'No'


class MainWindow(QMainWindow, Ui_MainWindow):
    # constructor
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        # a figure instance to plot on
        self.figure = plt.figure()

        # this is the Canvas Widget that
        # displays the 'figure'it takes the
        # 'figure' instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)

        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        self.toolbar = NavigationToolbar(self.canvas, self)

        # Just some button connected to 'plot' method
        self.button = QPushButton('Plot')

        # adding action to the button
        self.button.clicked.connect(self.plot)

        # creating a Vertical Box layout
        layout = QVBoxLayout()

        # adding tool bar to the layout
        layout.addWidget(self.toolbar)

        # adding canvas to the layout
        layout.addWidget(self.canvas)

        # adding push button to the layout
        layout.addWidget(self.button)

        # setting layout to the main window
        self.setLayout(layout)

    # action called by the push button
    def plot(self):
        # random data
        data = [random.random() for i in range(10)]

        # clearing old figure
        self.figure.clear()

        # create an axis
        ax = self.figure.add_subplot(111)

        # plot data
        ax.plot(data, '*-')

        # refresh canvas
        self.canvas.draw()



def main():
    app = QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())

if __name__ == '__main__':  # Если мы запускаем файл напрямую, а не импортируем
    main()  # то запускаем функцию main()

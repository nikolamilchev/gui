from PyQt5 import Qt
import pyqtgraph as pg
import numpy as np


class Window(Qt.QWidget):

    def __init__(self):
        super().__init__()

        layout = Qt.QVBoxLayout(self)

        self.view = view = pg.PlotWidget()
        self.curve = view.plot(name="Line")

        self.btn = Qt.QPushButton("Random plot")
        self.btn.clicked.connect(self.random_plot)

        layout.addWidget(Qt.QLabel("Some text"))
        layout.addWidget(self.view)
        layout.addWidget(self.btn)

    def random_plot(self):
        random_array = np.random.random_sample(20)
        self.curve.setData(random_array)


if __name__ == "__main__":
    app = Qt.QApplication([])
    w = Window()
    w.show()
    app.exec()
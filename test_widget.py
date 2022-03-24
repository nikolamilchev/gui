from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *


class Widget(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Tab example')

        # Обработка сигнала через метод
        self.button_next = QPushButton('Далее')
        self.button_next.clicked.connect(self._on_next_tab_clicked)

        # Обработка сигнала через лямбду
        self.button_prev = QPushButton('Назад')
        self.button_prev.clicked.connect(lambda: self.tab.setCurrentIndex(0))

        tab_1 = QFrame()
        layout_tab_1 = QVBoxLayout()
        layout_tab_1.addWidget(self.button_next)
        tab_1.setLayout(layout_tab_1)

        tab_2 = QFrame()
        layout_tab_2 = QVBoxLayout()
        layout_tab_2.addWidget(self.button_prev)
        tab_2.setLayout(layout_tab_2)

        self.tab = QTabWidget()
        self.tab.addTab(tab_1, "Основа")
        self.tab.addTab(tab_2, "Дополнительно")

        main_layout = QHBoxLayout()
        main_layout.addWidget(self.tab)
        self.setLayout(main_layout)

    def _on_next_tab_clicked(self):
        self.tab.setCurrentIndex(1)


if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)

    root = Widget()
    root.resize(400, 200)
    root.show()

    sys.exit(app.exec_())

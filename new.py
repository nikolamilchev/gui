import os
from math import sqrt, acos

from PyQt5 import QtCore, QtGui, QtWidgets, Qt
import pyqtgraph as pg
import sys
import design
import scipy.io
import threading
import numpy as np
import pandas as pd
from pyqtgraph import PlotWidget, plot
from PyQt5 import QtWidgets, uic
from pyqtgraph import PlotWidget, plot

import pyqtgraph as pg
import sys  # We need sys so that we can pass argv to QApplication
import os
import time

import sys
import matplotlib

matplotlib.use('Qt5Agg')

from PyQt5 import QtCore, QtGui, QtWidgets

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from PyQt5.QtWidgets import QApplication, QMainWindow, QGridLayout, QWidget, QTableWidget, QTableWidgetItem

from PyQt5 import QtCore



class PandasModel(QtCore.QAbstractTableModel):
    def __init__(self, df=pd.DataFrame(), parent=None):
        QtCore.QAbstractTableModel.__init__(self, parent=parent)
        self._df = df

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        if role != QtCore.Qt.DisplayRole:
            return QtCore.QVariant()

        if orientation == QtCore.Qt.Horizontal:
            try:
                return self._df.columns.tolist()[section]
            except (IndexError,):
                return QtCore.QVariant()
        elif orientation == QtCore.Qt.Vertical:
            try:
                # return self.df.index.tolist()
                return self._df.index.tolist()[section]
            except (IndexError,):
                return QtCore.QVariant()

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if role != QtCore.Qt.DisplayRole:
            return QtCore.QVariant()

        if not index.isValid():
            return QtCore.QVariant()

        return QtCore.QVariant(str(self._df.iloc[index.row(), index.column()]))

    def setData(self, index, value, role):
        row = self._df.index[index.row()]
        col = self._df.columns[index.column()]
        if hasattr(value, 'toPyObject'):
            # PyQt4 gets a QVariant
            value = value.toPyObject()
        else:
            # PySide gets an unicode
            dtype = self._df[col].dtype
            if dtype != object:
                value = None if value == '' else dtype.type(value)
        self._df.set_value(row, col, value)
        return True

    def rowCount(self, parent=QtCore.QModelIndex()):
        return len(self._df.index)

    def columnCount(self, parent=QtCore.QModelIndex()):
        return len(self._df.columns)

    def sort(self, column, order):
        colname = self._df.columns.tolist()[column]
        self.layoutAboutToBeChanged.emit()
        self._df.sort_values(colname, ascending=order == QtCore.Qt.AscendingOrder, inplace=True)
        self._df.reset_index(inplace=True, drop=True)
        self.layoutChanged.emit()



class ExampleApp(QtWidgets.QMainWindow):
    def __init__(self):
        # Это здесь нужно для доступа к переменным, методам
        # и т.д. в файле design.py
        super().__init__()
        self.time_index = 0
        uic.loadUi('data/design.ui', self)
        self.names_news = ['Сагиттальный наклон грудной клетки относительно пола',
                           'фронтальный наклон грудной клетки относительно пола',
                           'Сагиттальный наклон грудной клетки относительно таза',
                           'фронтальный наклон грудной клетки относительно таза',
                           'торсия груди относительно пола', 'торсия груди относительно таза',
                           'наклон головы относительно пола', 'наклон головы  относительно тела']
        self.names_old = ['APA', 'Dyn_Cobb', 'Dyn_LL', 'Dyn_PT', 'Dyn_SL', 'Dyn_SL_rotation', 'Dyn_TK']
        self.download_1()
        self.action.triggered.connect(self.download_1)
        self.action_2.triggered.connect(self.download_2)
        self.calc()
        self.parameter_plot.setBackground('w')
        self.comboBox.addItems(self.names_news)
        self.comboBox.addItems(self.names_old)
        self.comboBox.currentTextChanged.connect(self.mini_plots)
        self.time_index = 0
        self.mini_plots(value='Сагиттальный наклон грудной клетки относительно пола')
        self.horizontalSlider.valueChanged.connect(self.change_value_1)
        self.pushButton.clicked.connect(self.change_value)  # Выполнить вычисления
        self.calc_table()
        # Устанавливаем заголовки таблицы
        self.plot()

    def mini_plots(self, value):
        df = pd.concat([self.old_data, self.new_data])
        data_y = df.loc[value]
        data_x = range(len(data_y))
        self.parameter_plot.clear()
        self.parameter_plot.setBackground('w')
        self.parameter_plot.plot(data_x, data_y)
        self.parameter_plot.showGrid(x=True, y=True)

    def change_value_1(self, value):
        self.time_index = int(value)
        self.lcdNumber.display(int(value))

    def change_value(self, value):
        self.plot()
        self.calc_table()

    def plot(self):
        data_dynamic = self.data_[0]['Trajectories'][0][0]['Labeled'][0][0]['Data'][
            0]  # следующий индекс - номер маркера
        t = self.time_index
        x = data_dynamic[:, 0, t]
        y = data_dynamic[:, 2, t]
        self.sagital_widget.clear()
        self.sagital_widget.setBackground('w')
        self.sagital_widget.plot(x, y, pen=None, symbol='o', symbolSize=10)
        self.sagital_widget.showGrid(x=True, y=True)
        x = data_dynamic[:, 1, t]
        y = data_dynamic[:, 2, t]
        self.front_widget.clear()
        self.front_widget.setBackground('w')
        self.front_widget.plot(x, y, pen=None, symbol='o', symbolSize=10)
        self.front_widget.showGrid(x=True, y=True)

    def changeValue(self, value):  # Toolbar
        self.time_index = value

    def download_2(self):
        self.data_ = scipy.io.loadmat('data/walk 5km0001.mat')['walk_5km0001']
        self.horizontalSlider.setMaximum(2999)

    def download_1(self):
        self.data_ = scipy.io.loadmat('data/statica0001.mat')['statica0001']
        self.horizontalSlider.setMaximum(1726)

    def calc(self):
        data_dynamic = self.data_[0]['Trajectories'][0][0]['Labeled'][0][0]['Data'][
            0]  # следующий индекс - номер маркера
        param_1 = []
        vect_1 = data_dynamic[5][0:3] - data_dynamic[10][0:3]
        vect = np.array([1, 0, 0])
        for i in range(data_dynamic.shape[2]):
            param_1.append(np.dot(vect, vect_1[:, i]))
        vect_1 = data_dynamic[5][0:3] - data_dynamic[10][0:3]
        vect = np.array([0, 1, 0])
        param_2 = []
        for i in range(data_dynamic.shape[2]):
            param_2.append(np.dot(vect, vect_1[:, i]))
        v2 = (data_dynamic[12, 0:3, :] + data_dynamic[11, 0:3, :] + data_dynamic[14, 0:3, :] + data_dynamic[13, 0:3,
                                                                                               :]) / 4
        # Сагиттальный наклон грудной клетки относительно таза
        vect = [1, 0, 0]
        vect_1 = data_dynamic[5, 0:3] - v2
        param_3 = []
        for i in range(data_dynamic.shape[2]):
            param_3.append(np.dot(vect, vect_1[:, i]))

        # фронтальный наклон грудной клетки относительно таза
        vect = [0, 1, 0]
        vect_1 = data_dynamic[5, 0:3] - v2
        param_4 = []
        for i in range(data_dynamic.shape[2]):
            param_4.append(np.dot(vect, vect_1[:, i]))

        # торсия груди относительно пола
        vect = [1, 0, 0]
        vect_1 = data_dynamic[15, 0:3] - data_dynamic[10, 0:3]
        param_5 = []
        for i in range(data_dynamic.shape[2]):
            param_5.append(np.dot(vect, vect_1[:, i]))

        # торсия груди относительно таза
        vect = [1, 0, 0]
        vect_1 = data_dynamic[15, 0:3] - v2
        param_6 = []
        for i in range(data_dynamic.shape[2]):
            param_6.append(np.dot(vect, vect_1[:, i]))
        # наклон головы относительно пола
        vect = [1, 0, 0]
        vect_1 = data_dynamic[2, 0:3] - data_dynamic[10, 0:3]
        param_7 = []
        for i in range(data_dynamic.shape[2]):
            param_7.append(np.dot(vect, vect_1[:, i]))

        # наклон головы  относительно тела
        vect = [1, 0, 0]
        vect_1 = data_dynamic[2, 0:3] - v2
        param_8 = []
        for i in range(data_dynamic.shape[2]):
            param_8.append(np.dot(vect, vect_1[:, i]))

        # calculation Dyn-Cobb angle
        # calculation Dyn-Cobb angle: angle between T9-T11 p39-c in plane ZY
        l1 = np.array(
            (data_dynamic[5, 1] - data_dynamic[4, 1], data_dynamic[5, 2] - data_dynamic[4, 2]))  # vector p39-c
        l2 = np.array(
            (data_dynamic[7, 1] - data_dynamic[6, 1], data_dynamic[7, 2] - data_dynamic[6, 2]))  # vector T9-T11

        len_l1 = []
        len_l2 = []
        Dyn_Cobb = []
        for i in range(len(l1[0])):
            len_l1.append(sqrt(l1[0][i] ** 2 + l1[1][i] ** 2))
            len_l2.append(sqrt(l2[0][i] ** 2 + l2[1][i] ** 2))
            Dyn_Cobb.append(acos(np.dot(l1[:, i], l2[:, i]) / (len_l1[i] * len_l2[i])))

        # calculation Dyn-SL inclination : angle between R.ARC-L.ARC line Dyn-CVA(T1-S1) in plane ZY
        l1 = np.array(
            (data_dynamic[0, 1] - data_dynamic[1, 1], data_dynamic[0, 2] - data_dynamic[1, 2]))  # vector R.ARC-L.ARC
        l2 = np.array(
            (data_dynamic[2, 1] - data_dynamic[10, 1], data_dynamic[2, 2] - data_dynamic[10, 2]))  # vector T1-S1

        len_l1 = []
        len_l2 = []
        Dyn_SL = []
        for i in range(len(l1[0])):
            len_l1.append(sqrt(l1[0][i] ** 2 + l1[1][i] ** 2))
            len_l2.append(sqrt(l2[0][i] ** 2 + l2[1][i] ** 2))
            Dyn_SL.append(acos(np.dot(l1[:, i], l2[:, i]) / (len_l1[i] * len_l2[i])))

        # Dyn-TK angle: angle between T9-T11 p39-c in plane XZ
        l1 = np.array(
            (data_dynamic[4, 0] - data_dynamic[5, 0], data_dynamic[4, 2] - data_dynamic[5, 2]))  # vector p39-c
        l2 = np.array(
            (data_dynamic[6, 0] - data_dynamic[7, 0], data_dynamic[6, 2] - data_dynamic[7, 2]))  # vector T9-T11

        len_l1 = []
        len_l2 = []
        Dyn_TK = []
        for i in range(len(l1[0])):
            len_l1.append(sqrt(l1[0][i] ** 2 + l1[1][i] ** 2))
            len_l2.append(sqrt(l2[0][i] ** 2 + l2[1][i] ** 2))
            Dyn_TK.append(acos(np.dot(l1[:, i], l2[:, i]) / (len_l1[i] * len_l2[i])))

        # calculation Dyn-LL angle:   angle between T11-p114 L4-p114 in plane ZX
        l1 = np.array(
            (data_dynamic[9, 0] - data_dynamic[8, 0], data_dynamic[9, 2] - data_dynamic[8, 2]))  # vector L4-p114
        l2 = np.array(
            (data_dynamic[8, 0] - data_dynamic[7, 0], data_dynamic[8, 2] - data_dynamic[9, 2]))  # vector T11-p114

        len_l1 = []
        len_l2 = []
        Dyn_LL = []
        for i in range(len(l1[0])):
            len_l1.append(sqrt(l1[0][i] ** 2 + l1[1][i] ** 2))
            len_l2.append(sqrt(l2[0][i] ** 2 + l2[1][i] ** 2))
            Dyn_LL.append(acos(np.dot(l1[:, i], l2[:, i]) / (len_l1[i] * len_l2[i])))

        # calculation  Dyn-PT two ways
        # calculation Dyn-PT : angle between L.PSIS-L.ASIS and normal vector к ZY in plane ZX
        l1 = np.array(
            (data_dynamic[11, 0] - data_dynamic[13, 1],
             data_dynamic[11, 2] - data_dynamic[13, 2]))  # vector L.PSIS-L.ASIS
        l2 = np.array(
            (data_dynamic[11, 0] - data_dynamic[13, 1], data_dynamic[11, 2] - data_dynamic[1, 2]))  # normal vector

        len_l1 = []
        len_l2 = []

        Dyn_PT = []
        for i in range(len(l1[0])):
            len_l1.append(sqrt(l1[0][i] ** 2 + l1[1][i] ** 2))
            len_l2.append(sqrt(l2[0][i] ** 2 + l2[1][i] ** 2))
            Dyn_PT.append(acos(np.dot(l1[:, i], l2[:, i]) / (len_l1[i] * len_l2[i])))
        # calculation Dyn-PT : angle between R.PSIS-R.ASIS and normal vector к ZY in plane ZX
        l1 = np.array(
            (data_dynamic[12, 0] - data_dynamic[14, 0],
             data_dynamic[12, 2] - data_dynamic[14, 2]))  # vector L.PSIS-L.ASIS
        l2 = np.array(
            (data_dynamic[12, 0] - data_dynamic[14, 0], data_dynamic[12, 2] - data_dynamic[12, 2]))  # normal    vector

        len_l1 = []
        len_l2 = []

        Dyn_PT = []
        for i in range(len(l1[0])):
            len_l1.append(sqrt(l1[0][i] ** 2 + l1[1][i] ** 2))
            len_l2.append(sqrt(l2[0][i] ** 2 + l2[1][i] ** 2))
            Dyn_PT.append(acos(np.dot(l1[:, i], l2[:, i]) / (len_l1[i] * len_l2[i])))
        # calculation Dyn_SL_rotation : angle between R.ARC-L.ARC and normal vector of ZX in plane XY\
        l2 = np.array(
            (data_dynamic[12, 0] - data_dynamic[13, 0],
             data_dynamic[12, 2] - data_dynamic[13, 2]))  # vector R.ASIS-L.ASIS

        l1 = []
        for i in range(len(l2[0])):
            l1.append(np.array([0, -1]))  # normal vector

        len_l1 = []
        len_l2 = []

        Dyn_SL_rotation = []
        for i in range(len(l2[0])):
            len_l1.append(sqrt(l1[i][0] ** 2 + l1[i][1] ** 2))
            len_l2.append(sqrt(l2[0][i] ** 2 + l2[1][i] ** 2))
            Dyn_SL_rotation.append(acos(np.dot(l1[i], l2[:, i]) / (len_l1[i] * len_l2[i])))
        # calculation pelvic_rotation : angle between R.ASIS-L.ASIS  and normal vector of ZX in plane XY
        l1 = np.array(
            (data_dynamic[1, 0] - data_dynamic[0, 0], data_dynamic[1, 2] - data_dynamic[0, 2]))  # vector L.PSIS-L.ASIS

        l2 = []
        for i in range(len(l1[0])):
            l2.append(np.array([0, 1]))  # normal vector

        len_l1 = []
        len_l2 = []

        pelvic_rotation = []
        for i in range(len(l1[0])):
            len_l1.append(sqrt(l1[0][i] ** 2 + l1[1][i] ** 2))
            len_l2.append(sqrt(l2[i][0] ** 2 + l2[i][1] ** 2))
            pelvic_rotation.append(acos(np.dot(l1[:, i], l2[i]) / (len_l1[i] * len_l2[i])))
            # calculation APA : angle between R.ARC-L.ARC and R.ASIS-L.ASIS in plane XY
        l1 = np.array(
            (data_dynamic[1, 0] - data_dynamic[0, 0], data_dynamic[1, 2] - data_dynamic[0, 2]))  # vector L.PSIS-L.ASIS
        l2 = np.array(
            (data_dynamic[13, 0] - data_dynamic[14, 0],
             data_dynamic[13, 2] - data_dynamic[14, 2]))  # vector R.ASIS-L.ASIS

        len_l1 = []
        len_l2 = []

        APA = []
        for i in range(len(l1[0])):
            len_l1.append(sqrt(l1[0][i] ** 2 + l1[1][i] ** 2))
            len_l2.append(sqrt(l2[0][i] ** 2 + l2[1][i] ** 2))
            APA.append(acos(np.dot(l1[:, i], l2[:, i]) / (len_l1[i] * len_l2[i])))
        data_1 = pd.DataFrame([param_1, param_2, param_3, param_4, param_5, param_6, param_7, param_8],
                              index=self.names_news)
        data_2 = pd.DataFrame([APA, Dyn_Cobb, Dyn_LL, Dyn_PT, Dyn_SL, Dyn_SL_rotation, Dyn_TK],
                              index=self.names_old)
        self.old_data = data_1
        self.new_data = data_2

    def calc_table(self):
        self.calc()
        data_1, data_2 = self.old_data, self.new_data
        self.show_table(data_1, data_2)

    def show_table(self, data_1, data_2):
        model_1 = PandasModel(data_1[self.time_index].apply(lambda x: round(x, 8)).to_frame(name="time " + str(self.time_index)))
        model_2 = PandasModel(data_2[self.time_index].apply(lambda x: round(x, 8)).to_frame(name="time " + str(self.time_index)))
        self.tableView.setModel(model_1)
        self.tableView_2.setModel(model_2)
        # переделать вывод использовать тул бар для вывода отрисовать графики


def main():
    app = QtWidgets.QApplication(sys.argv)  # Новый экземпляр QApplication
    window = ExampleApp()  # Создаём объект класса ExampleApp
    window.show()  # Показываем окно
    app.exec_()  # и запускаем приложение


if __name__ == '__main__':  # Если мы запускаем файл напрямую, а не импортируем
    main()  # то запускаем функцию main()

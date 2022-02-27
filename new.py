import os
from math import sqrt, acos

from PyQt5 import QtCore, QtGui, QtWidgets, Qt
import sys
import design
import scipy.io
import numpy as np
import pandas as pd

from PyQt5 import QtWidgets, uic
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg

import pyqtgraph as pg
import sys  # We need sys so that we can pass argv to QApplication
import os

import sys
import matplotlib
matplotlib.use('Qt5Agg')

from PyQt5 import QtCore, QtGui, QtWidgets

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from PyQt5.QtWidgets import QApplication, QMainWindow, QGridLayout, QWidget, QTableWidget, QTableWidgetItem
from PyQt5.QtCore import QSize, Qt



class ExampleApp(QtWidgets.QMainWindow, design.Ui_MainWindow):
    def __init__(self):
        # Это здесь нужно для доступа к переменным, методам
        # и т.д. в файле design.py
        super().__init__()
        self.time_index = 0

        self.setupUi(self)  # Это нужно для инициализации нашего дизайна
        self.static_data = None
        self.move_data = None
        self.download()
        self.pushButton.clicked.connect(self.calc)  # Выполнить функцию browse_folder
        # Устанавливаем заголовки таблицы
        sld = self.horizontalSlider



        sld.valueChanged[int].connect(self.changeValue)
        data = pd.DataFrame([[]])
        self.plot_(data)
    def changeValue(self, value):
        self.time_index = value
        self.lineEdit.setText(str(value))
    def download(self):
        self.static_data  = scipy.io.loadmat('walk 5km0001.mat')
        self.move_data = scipy.io.loadmat('statica0001.mat')

    def calc(self):
        walk_5km0001 = self.static_data
        statica0001 = self.move_data
        data_dynamic = walk_5km0001['walk_5km0001'][0]['Trajectories'][0][0]['Labeled'][0][0]['Data'][
            0]  # следующий индекс - номер маркера
        data_static = statica0001['statica0001'][0]['Trajectories'][0][0]['Labeled'][0][0]['Data'][
            0]  # следующий индекс - номер маркера
        param_1 = []
        vect_1 = data_dynamic[5][0:3] - data_dynamic[10][0:3]
        vect = np.array([1, 0, 0])
        for i in range(3000):
            param_1.append(np.dot(vect, vect_1[:, i]))
        vect_1 = data_dynamic[5][0:3] - data_dynamic[10][0:3]
        vect = np.array([0, 1, 0])
        param_2 = []
        for i in range(3000):
            param_2.append(np.dot(vect, vect_1[:, i]))
        v2 = (data_dynamic[12, 0:3, :] + data_dynamic[11, 0:3, :] + data_dynamic[14, 0:3, :] + data_dynamic[13, 0:3,
                                                                                               :]) / 4
        # Сагиттальный наклон грудной клетки относительно таза
        vect = [1, 0, 0]
        vect_1 = data_dynamic[5, 0:3] - v2
        param_3 = []
        for i in range(3000):
            param_3.append(np.dot(vect, vect_1[:, i]))

        param_3
        # фронтальный наклон грудной клетки относительно таза
        vect = [0, 1, 0]
        vect_1 = data_dynamic[5, 0:3] - v2
        param_4 = []
        for i in range(3000):
            param_4.append(np.dot(vect, vect_1[:, i]))

        param_4
        # торсия груди относительно пола
        vect = [1, 0, 0]
        vect_1 = data_dynamic[15, 0:3] - data_dynamic[10, 0:3]
        param_5 = []
        for i in range(3000):
            param_5.append(np.dot(vect, vect_1[:, i]))

        param_5
        # торсия груди относительно таза
        vect = [1, 0, 0]
        vect_1 = data_dynamic[15, 0:3] - v2
        param_6 = []
        for i in range(3000):
            param_6.append(np.dot(vect, vect_1[:, i]))

        param_6
        # наклон головы относительно пола
        vect = [1, 0, 0]
        vect_1 = data_dynamic[2, 0:3] - data_dynamic[10, 0:3]
        param_7 = []
        for i in range(3000):
            param_7.append(np.dot(vect, vect_1[:, i]))

        param_7
        # наклон головы  относительно тела
        vect = [1, 0, 0]
        vect_1 = data_dynamic[2, 0:3] - v2
        param_8 = []
        for i in range(3000):
            param_8.append(np.dot(vect, vect_1[:, i]))

        # calculation Dyn-Cobb angle
        # calculation Dyn-Cobb angle: angle between T9-T11 p39-c in plane ZY
        l1 = np.array((data_static[5, 1] - data_static[4, 1], data_static[5, 2] - data_static[4, 2]))  # vector p39-c
        l2 = np.array((data_static[7, 1] - data_static[6, 1], data_static[7, 2] - data_static[6, 2]))  # vector T9-T11

        len_l1 = []
        len_l2 = []
        Dyn_Cobb = []
        for i in range(len(l1[0])):
            len_l1.append(sqrt(l1[0][i] ** 2 + l1[1][i] ** 2))
            len_l2.append(sqrt(l2[0][i] ** 2 + l2[1][i] ** 2))
            Dyn_Cobb.append(acos(np.dot(l1[:, i], l2[:, i]) / (len_l1[i] * len_l2[i])))

        # calculation Dyn-SL inclination : angle between R.ARC-L.ARC line Dyn-CVA(T1-S1) in plane ZY
        l1 = np.array(
            (data_static[0, 1] - data_static[1, 1], data_static[0, 2] - data_static[1, 2]))  # vector R.ARC-L.ARC
        l2 = np.array((data_static[2, 1] - data_static[10, 1], data_static[2, 2] - data_static[10, 2]))  # vector T1-S1

        len_l1 = []
        len_l2 = []
        Dyn_SL = []
        for i in range(len(l1[0])):
            len_l1.append(sqrt(l1[0][i] ** 2 + l1[1][i] ** 2))
            len_l2.append(sqrt(l2[0][i] ** 2 + l2[1][i] ** 2))
            Dyn_SL.append(acos(np.dot(l1[:, i], l2[:, i]) / (len_l1[i] * len_l2[i])))

        # Dyn-TK angle: angle between T9-T11 p39-c in plane XZ
        l1 = np.array((data_static[4, 0] - data_static[5, 0], data_static[4, 2] - data_static[5, 2]))  # vector p39-c
        l2 = np.array((data_static[6, 0] - data_static[7, 0], data_static[6, 2] - data_static[7, 2]))  # vector T9-T11

        len_l1 = []
        len_l2 = []
        Dyn_TK = []
        for i in range(len(l1[0])):
            len_l1.append(sqrt(l1[0][i] ** 2 + l1[1][i] ** 2))
            len_l2.append(sqrt(l2[0][i] ** 2 + l2[1][i] ** 2))
            Dyn_TK.append(acos(np.dot(l1[:, i], l2[:, i]) / (len_l1[i] * len_l2[i])))

        # calculation Dyn-LL angle:   angle between T11-p114 L4-p114 in plane ZX
        l1 = np.array((data_static[9, 0] - data_static[8, 0], data_static[9, 2] - data_static[8, 2]))  # vector L4-p114
        l2 = np.array((data_static[8, 0] - data_static[7, 0], data_static[8, 2] - data_static[9, 2]))  # vector T11-p114

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
            (data_static[11, 0] - data_static[13, 1], data_static[11, 2] - data_static[13, 2]))  # vector L.PSIS-L.ASIS
        l2 = np.array(
            (data_static[11, 0] - data_static[13, 1], data_static[11, 2] - data_static[1, 2]))  # normal vector

        len_l1 = []
        len_l2 = []

        Dyn_PT = []
        for i in range(len(l1[0])):
            len_l1.append(sqrt(l1[0][i] ** 2 + l1[1][i] ** 2))
            len_l2.append(sqrt(l2[0][i] ** 2 + l2[1][i] ** 2))
            Dyn_PT.append(acos(np.dot(l1[:, i], l2[:, i]) / (len_l1[i] * len_l2[i])))
        # calculation Dyn-PT : angle between R.PSIS-R.ASIS and normal vector к ZY in plane ZX
        l1 = np.array(
            (data_static[12, 0] - data_static[14, 0], data_static[12, 2] - data_static[14, 2]))  # vector L.PSIS-L.ASIS
        l2 = np.array(
            (data_static[12, 0] - data_static[14, 0], data_static[12, 2] - data_static[12, 2]))  # normal    vector

        len_l1 = []
        len_l2 = []

        Dyn_PT = []
        for i in range(len(l1[0])):
            len_l1.append(sqrt(l1[0][i] ** 2 + l1[1][i] ** 2))
            len_l2.append(sqrt(l2[0][i] ** 2 + l2[1][i] ** 2))
            Dyn_PT.append(acos(np.dot(l1[:, i], l2[:, i]) / (len_l1[i] * len_l2[i])))
        # calculation Dyn_SL_rotation : angle between R.ARC-L.ARC and normal vector of ZX in plane XY\
        l2 = np.array(
            (data_static[12, 0] - data_static[13, 0], data_static[12, 2] - data_static[13, 2]))  # vector R.ASIS-L.ASIS

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
            (data_static[1, 0] - data_static[0, 0], data_static[1, 2] - data_static[0, 2]))  # vector L.PSIS-L.ASIS

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
            (data_static[1, 0] - data_static[0, 0], data_static[1, 2] - data_static[0, 2]))  # vector L.PSIS-L.ASIS
        l2 = np.array(
            (data_static[13, 0] - data_static[14, 0], data_static[13, 2] - data_static[14, 2]))  # vector R.ASIS-L.ASIS

        len_l1 = []
        len_l2 = []

        APA = []
        for i in range(len(l1[0])):
            len_l1.append(sqrt(l1[0][i] ** 2 + l1[1][i] ** 2))
            len_l2.append(sqrt(l2[0][i] ** 2 + l2[1][i] ** 2))
            APA.append(acos(np.dot(l1[:, i], l2[:, i]) / (len_l1[i] * len_l2[i])))
        data = pd.DataFrame([APA,Dyn_Cobb,Dyn_LL,Dyn_PT,Dyn_SL,Dyn_SL_rotation,Dyn_TK],index = ['APA','Dyn_Cobb','Dyn_LL','Dyn_PT','Dyn_SL','Dyn_SL_rotation','Dyn_TK'])
        print(data)
        model = self.table_f(data)
        #переделать вывод использовать тул бар для вывода отрисовать графики
    def table_f(self,data):
        central_widget = self.tableWidget                # Создаём центральный виджет
        print('here')
        i = 2
        print('here')
        table = central_widget  # Создаём таблицу
        print('here')
        print('here')
        headers = data.columns.values.tolist()
        print('here')
        print('here')
        #table.setHorizontalHeaderLabels(headers)
        print('here')
        print('here')
        for i, row in data.iterrows():
            # Добавление строки
            table.setRowCount(table.rowCount() + 1)

            for j in range(2):
                table.setItem(i, j, QTableWidgetItem(str(row[j])))

        table.show()
    def plot_(self,data):

        self.graphWidget = pg.PlotWidget()

        hour = range(data.shape[0])
        temperature = data.loc[0]

        # plot data: x, y values
        self.graphWidget.plot(hour, temperature)



class PandasModel(QtCore.QAbstractTableModel):
    def __init__(self, data, parent=None):
        QtCore.QAbstractTableModel.__init__(self, parent)
        self._data = data
    def rowCount(self, parent=None):
        return len(self._data.values)
    def columnCount(self, parent=None):
        return self._data.columns.size
    def data(self, index, role=QtCore.Qt.DisplayRole):
        if index.isValid():
            if role == QtCore.Qt.DisplayRole:
                return QtCore.QVariant(str(
                    self._data.iloc[index.row()][index.column()]))
        return QtCore.QVariant()



def main():
    app = QtWidgets.QApplication(sys.argv)  # Новый экземпляр QApplication
    window = ExampleApp()  # Создаём объект класса ExampleApp
    window.show()  # Показываем окно
    app.exec_()  # и запускаем приложение


if __name__ == '__main__':  # Если мы запускаем файл напрямую, а не импортируем
    main()  # то запускаем функцию main()


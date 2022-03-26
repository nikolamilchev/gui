from math import sqrt, acos
import scipy.io
import numpy as np
import pandas as pd
from PyQt5 import uic

import matplotlib
import matplotlib.pyplot as plt
import pyqtgraph
import sys

from PyQt5.QtCore import QFile, Qt
from PyQt5.QtWidgets import QFileDialog

matplotlib.use('Qt5Agg')
import os

from PyQt5 import QtWidgets

from PyQt5 import QtCore

import design_08 as design

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

from pyqtgraph import PlotWidget

class ExampleApp(QtWidgets.QMainWindow,design.Ui_MainWindow):
    def __init__(self):
        # Это здесь нужно для доступа к переменным, методам
        # и т.д. в файле design.py
        super().__init__()
        self.setupUi(self)
        self.time_index = 0
        self.data_y_min_1 = None
        self.data_x_min_1 = None
        self.data_y_min_2 = None
        self.data_x_min_2 = None
        self.name_mini_plot = None
        self.data_1 = None
        self.data_2 = None

        self.dimension = {'Вариант 1. сагиттальная плоскость': 'смещение, мм',
                           'Вариант 1. фронтальная плоскость': 'смещение, мм',
                           'Вариант 2. сагиттальная плоскость': 'смещение, мм',
                           'Вариант 2. фронтальная плоскость': 'смещение, мм',
                           'торсия груди относительно пола': 'смещение, мм', 'торсия груди относительно таза' : 'градус угла',
                           'наклон головы относительно пола' : 'градус угла', 'наклон головы  относительно тела': 'градус угла',
                           'Сагиттальный наклон грудной клетки относительно пола': 'градус угла',
                           'фронтальный наклон грудной клетки относительно пола': 'градус угла',
                           'Сагиттальный наклон грудной клетки относительно таза': 'градус угла',
                           'фронтальный наклон грудной клетки относительно таза': 'градус угла',
                          'acromion pelvis angle': 'градус угла', 'Cobb angle': 'градус угла', 'lumbar lordosis': 'градус угла', 'pelvic tilt': 'градус угла',
                          'shoulder line inclination': 'градус угла', 'shoulder line rotation': 'градус угла', 'thoracic kyphosis': 'градус угла'}
        self.names_news = ['Вариант 1. сагиттальная плоскость',
                           'Вариант 1. фронтальная плоскость',
                           'Вариант 2. сагиттальная плоскость',
                           'Вариант 2. фронтальная плоскость',
                           'торсия груди относительно пола', 'торсия груди относительно таза',
                           'наклон головы относительно пола', 'наклон головы  относительно тела',
                           'Сагиттальный наклон грудной клетки относительно пола',
                           'фронтальный наклон грудной клетки относительно пола',
                           'Сагиттальный наклон грудной клетки относительно таза',
                           'фронтальный наклон грудной клетки относительно таза']
        self.names_old = ['acromion pelvis angle', 'Cobb angle', 'lumbar lordosis', 'pelvic tilt', 'shoulder line inclination', 'shoulder line rotation', 'thoracic kyphosis']

        self.action.triggered.connect(self.download_1)
        self.action_2.triggered.connect(self.download_2)
        self.parameter_plot.setBackground('w')
        self.comboBox.addItems(self.names_news)
        self.comboBox.addItems(self.names_old)
        self.comboBox.currentTextChanged.connect(self.mini_plots)
        self.time_index = 1
        self.old_data = None
        self.new_data = None
        self.horizontalSlider.valueChanged.connect(self.change_value_1)
        self.pushButton.clicked.connect(self.change_value)  # Выполнить вычисления
        self.pushButton_3.clicked.connect(self.download_file)
        self.pushButton_2.clicked.connect(self.save_to_png)
        self.checkBox.stateChanged.connect(self.download)
        # Устанавливаем заголовки таблицы

    def download(self,state):
        if state == Qt.Checked:
            self.download_1()
        else:
            self.download_2()

    def download_file(self):
        T = True
        while T:
            filename = QFileDialog.getOpenFileName(filter ='*.mat')
            path = filename[0]
            try:
                self.data_1 = scipy.io.loadmat(path)['walk_5km0001']
            except KeyError:
                self.data_2 = scipy.io.loadmat(path)['statica0001']

            filename = QFileDialog.getOpenFileName(filter ='*.mat')
            path = filename[0]
            try:
                self.data_1 = scipy.io.loadmat(path)['walk_5km0001']
            except KeyError:
                self.data_2 = scipy.io.loadmat(path)['statica0001']
            if self.data_1 is None or self.data_2 is None:
                T = True
            else:
                T = False
        #Включение графиков
        self.showMaximized()
        self.showNormal()
        self.data_ = self.data_1
        self.change_value(None)
        self.mini_plots(value= self.names_news[0])





    def save_to_png(self):
        fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
        if os.path.exists('./plot/static') ==False:
            os.makedirs('plot/static')
        if os.path.exists('./plot/dynamic') ==False:
            os.makedirs('plot/dynamic')
        ax.plot(self.data_x_min_1, self.data_y_min_1)
        ax.set_title(self.name_mini_plot)
        ax.set_xlabel('Цикл шага (%)')
        ax.set_ylabel(self.dimension[self.name_mini_plot]) #добавить мм для баланса
        if self.checkBox.isChecked():
            fig.savefig('plot/static/'+self.name_mini_plot+'.png')  # save the figure to file
        else:
            fig.savefig('plot/dynamic/'+self.name_mini_plot+'.png')  # save the figure to file
        plt.close(fig)
    def mini_plots(self, value):
        self.name_mini_plot = value
        df = pd.concat([self.old_data, self.new_data])
        data_y = df.loc[self.name_mini_plot]
        data_x = [i for i in range(len(data_y))]
        data_x = [i/max(data_x)*100 for i in data_x]
        self.data_y_min_1 = data_y
        self.data_x_min_1 = data_x
        self.data_y_min_2 = data_y
        self.data_x_min_2 = data_x
        self.mini_plots_1(data_y,data_x)
        self.mini_plots_2(data_y,data_x)

    def mini_plots_1(self, data_y,data_x):
        self.parameter_plot.clear()
        self.parameter_plot.setBackground('w')
        self.parameter_plot.plot(data_x, data_y)
        self.parameter_plot.showGrid(x=True, y=True)

    def mini_plots_2(self, data_y,data_x):
        self.mean_plot.clear()
        self.mean_plot.setBackground('w')
        self.mean_plot.plot(data_x, data_y)
        self.mean_plot.showGrid(x=True, y=True)

    def change_value_1(self, value):
        self.time_index = int(value)
        self.lcdNumber.display(int(value))

    def change_value(self, value):
        self.calc_table()
        self.plot()

    def plot(self):
        data_dynamic = self.data_[0]['Trajectories'][0][0]['Labeled'][0][0]['Data'][
            0]  # следующий индекс - номер маркера
        t = self.time_index
        x = data_dynamic[:, 0, t]
        y = data_dynamic[:, 2, t]
        self.sagital_widget.clear()
        self.sagital_widget.setBackground('w')
        self.sagital_widget.plot(x, y, pen=None, symbol='o', symbolSize=7)
        self.sagital_widget.showGrid(x=True, y=True)
        x = data_dynamic[:, 1, t]
        y = data_dynamic[:, 2, t]
        self.front_widget.clear()
        self.front_widget.setBackground('w')
        self.front_widget.plot(x, y, pen=None, symbol='o', symbolSize=7)
        self.front_widget.showGrid(x=True, y=True)

    def changeValue(self, value):  # Toolbar
        self.time_index = value

    def download_2(self):
        if self.data_1 is not None:
            self.data_ = self.data_1
            self.horizontalSlider.setMaximum(self.data_[0]['Trajectories'][0][0]['Labeled'][0][0]['Data'][
            0].shape[2]-1)
            self.calc_table()
            self.mini_plots(self.comboBox.currentText())
            self.plot()

    def download_1(self):
        if self.data_2 is not None:
            self.data_ = self.data_2
            self.horizontalSlider.setMaximum(self.data_[0]['Trajectories'][0][0]['Labeled'][0][0]['Data'][
            0].shape[2]-1)
            self.calc_table()
            self.mini_plots(self.comboBox.currentText())
            self.plot()

    def calc(self):
        data_dynamic = self.data_[0]['Trajectories'][0][0]['Labeled'][0][0]['Data'][
            0]  # следующий индекс - номер маркера
        v2 = (data_dynamic[12, 0:3, :] + data_dynamic[11, 0:3, :] + data_dynamic[14, 0:3, :] + data_dynamic[13, 0:3,
                                                                                               :]) / 4
        param_1 = self.param_1(data_dynamic)
        param_2 = self.param_2(data_dynamic)
        param_3 = self.param_3(data_dynamic,v2)
        param_4 = self.param_4(data_dynamic,v2)
        param_5 = self.param_5(data_dynamic)
        param_6 = self.param_6(data_dynamic,v2)
        param_7 = self.param_7(data_dynamic)
        param_8 = self.param_8(data_dynamic,v2)
        param_9 = self.param_9(data_dynamic)
        param_10 = self.param_10(data_dynamic)
        param_11 = self.param_11(data_dynamic,v2)
        param_12 = self.param_12(data_dynamic,v2)

        APA = self.APA(data_dynamic)
        Dyn_Cobb = self.Dyn_Cobb(data_dynamic)
        Dyn_LL = self.Dyn_LL(data_dynamic)
        Dyn_PT = self.Dyn_PT(data_dynamic)
        Dyn_SL = self.Dyn_SL(data_dynamic)
        Dyn_SL_rotation = self.Dyn_SL_rotation(data_dynamic)
        Dyn_TK = self.Dyn_TK(data_dynamic)


        data_1 = pd.DataFrame([param_1, param_2, param_3, param_4, param_5, param_6, param_7, param_8, param_9, param_10, param_11, param_12],
                              index=self.names_news)
        data_2 = pd.DataFrame([APA, Dyn_Cobb, Dyn_LL, Dyn_PT, Dyn_SL, Dyn_SL_rotation, Dyn_TK],
                              index=self.names_old)
        self.old_data = data_1
        self.new_data = data_2
    def LHS(self, data_dynamic):
        # здесь будет происходит деление
        print(data_dynamic[13])
        RHS = []

        return RHS

    def RHS(self, data_dynamic):
        # здесь будет происходит деление
        print(data_dynamic[14])
        LHS = []
        for i in range(1,data_dynamic[14]):
            pass

        return LHS


    def param_9(self, data_dynamic):
        # Сагиттальный наклон грудной клетки относительно пола
        param_1 = []
        vect_1 = data_dynamic[2][0:3] - data_dynamic[10][0:3]
        vect = np.array([1, 0, 0])
        for i in range(data_dynamic.shape[2]):
            param_1.append(np.dot(vect, vect_1[:, i]))
        return param_1

    def param_10(self, data_dynamic):
        # фронтальный наклон грудной клетки относительно пола
        vect_1 = data_dynamic[2][0:3] - data_dynamic[10][0:3]
        vect = np.array([0, 1, 0])
        param_2 = []
        for i in range(data_dynamic.shape[2]):
            param_2.append(np.dot(vect, vect_1[:, i]))
        return param_2

    def param_11(self, data_dynamic,v2):
        # Сагиттальный наклон грудной клетки относительно таза
        vect = [1, 0, 0]
        vect_1 = data_dynamic[2, 0:3] - v2
        param_3 = []
        for i in range(data_dynamic.shape[2]):
            param_3.append(np.dot(vect, vect_1[:, i]))
        return param_3
    def param_12(self,data_dynamic,v2):
        # фронтальный наклон грудной клетки относительно таза
        vect = [0, 1, 0]
        vect_1 = data_dynamic[2, 0:3] - v2
        param_4 = []
        for i in range(data_dynamic.shape[2]):
            param_4.append(np.dot(vect, vect_1[:, i]))
        return param_4
    def param_1(self, data_dynamic):
        #'Вариант 1. сагиттальная плоскость'
        param_1 = []
        for i in range(data_dynamic.shape[2]):
            param_1.append(np.linalg.norm(data_dynamic[10][0:3:2,i] - data_dynamic[2][0:3:2,i]))
        return param_1

    def param_2(self, data_dynamic):
        #  Вариант 1. фронтальная плоскость
        param_2 = []
        for i in range(data_dynamic.shape[2]):
            param_2.append(np.linalg.norm(data_dynamic[10][1:3,i] - data_dynamic[2][1:3,i]))
        return param_2

    def param_3(self, data_dynamic,v2):
        #         'Вариант 2. сагиттальная плоскость',
        param_3 = []
        for i in range(data_dynamic.shape[2]):
            param_3.append(np.linalg.norm(data_dynamic[2][0:3:2,i] - v2[0:3:2,i]))
        return param_3
    def param_4(self,data_dynamic,v2):
        #'Вариант 2. фронтальная плоскость',
        param_4 = []
        for i in range(data_dynamic.shape[2]):
            param_4.append(np.linalg.norm(data_dynamic[2][1:3,i] - v2[1:3,i]))
        return param_4
    def param_5(self,data_dynamic):
        # торсия груди относительно пола
        vect = [1, 0, 0]
        vect_1 = data_dynamic[15, 0:3] - data_dynamic[10, 0:3]
        param_5 = []
        for i in range(data_dynamic.shape[2]):
            param_5.append(np.dot(vect, vect_1[:, i]))
        return param_5
    def param_6(self,data_dynamic,v2):
        # торсия груди относительно таза
        vect = [1, 0, 0]
        vect_1 = data_dynamic[15, 0:3] - v2
        param_6 = []
        for i in range(data_dynamic.shape[2]):
            param_6.append(np.dot(vect, vect_1[:, i]))
        return param_6
    def param_7(self,data_dynamic):
        # наклон головы относительно пола
        vect = [1, 0, 0]
        vect_1 = data_dynamic[2, 0:3] - data_dynamic[10, 0:3]
        param_7 = []
        for i in range(data_dynamic.shape[2]):
            param_7.append(np.dot(vect, vect_1[:, i]))
        return param_7

    def param_8(self, data_dynamic,v2):
        # наклон головы  относительно тела
        vect = [1, 0, 0]
        vect_1 = data_dynamic[2, 0:3] - v2
        param_8 = []
        for i in range(data_dynamic.shape[2]):
            param_8.append(np.dot(vect, vect_1[:, i]))
        return param_8
    def Dyn_Cobb(self,data_dynamic):
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
        return Dyn_Cobb

    def Dyn_SL(self,data_dynamic):
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
        return Dyn_SL

    def Dyn_TK(self,data_dynamic):
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
        return Dyn_TK
    def Dyn_LL(self,data_dynamic):
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
        return Dyn_LL

    def Dyn_PT(self,data_dynamic):
        # calculation  Dyn-PT two ways
        # calculation Dyn-PT : angle between L.PSIS-L.ASIS and normal vector к ZY in plane ZX
        l1 = np.array(
            (data_dynamic[11, 0] - data_dynamic[13, 1],
             data_dynamic[11, 2] - data_dynamic[13, 2]))  # vector L.PSIS-L.ASIS
        l2 = np.array(
            (data_dynamic[11, 0] - data_dynamic[13, 1], data_dynamic[11, 2] - data_dynamic[1, 2]))  # normal vector

        len_l1 = []
        len_l2 = []

        Dyn_PT_1 = []
        for i in range(len(l1[0])):
            len_l1.append(sqrt(l1[0][i] ** 2 + l1[1][i] ** 2))
            len_l2.append(sqrt(l2[0][i] ** 2 + l2[1][i] ** 2))
            Dyn_PT_1.append(acos(np.dot(l1[:, i], l2[:, i]) / (len_l1[i] * len_l2[i])))
        # calculation Dyn-PT : angle between R.PSIS-R.ASIS and normal vector к ZY in plane ZX
        l1 = np.array(
            (data_dynamic[12, 0] - data_dynamic[14, 0],
             data_dynamic[12, 2] - data_dynamic[14, 2]))  # vector L.PSIS-L.ASIS
        l2 = np.array(
            (data_dynamic[12, 0] - data_dynamic[14, 0], data_dynamic[12, 2] - data_dynamic[12, 2]))  # normal    vector

        len_l1 = []
        len_l2 = []

        Dyn_PT_2 = []
        for i in range(len(l1[0])):
            len_l1.append(sqrt(l1[0][i] ** 2 + l1[1][i] ** 2))
            len_l2.append(sqrt(l2[0][i] ** 2 + l2[1][i] ** 2))
            Dyn_PT_2.append(acos(np.dot(l1[:, i], l2[:, i]) / (len_l1[i] * len_l2[i])))
        return Dyn_PT_1

    def Dyn_SL_rotation(self,data_dynamic):

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
        return Dyn_SL_rotation

    def pelvic_rotation(self,data_dynamic):
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
        return pelvic_rotation

    def APA(self,data_dynamic):
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
        return APA


    def calc_table(self):
        self.calc()
        self.show_table()

    def show_table(self):
        model_1 = PandasModel(
            self.old_data[self.time_index].apply(lambda x: round(x, 8)).to_frame(name="time " + str(self.time_index)))
        model_2 = PandasModel(
            self.new_data[self.time_index].apply(lambda x: round(x, 8)).to_frame(name="time " + str(self.time_index)))
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

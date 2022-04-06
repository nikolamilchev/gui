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

import design_095 as design

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
        # Устанавливаем заголовки таблицы


    def download_file(self):
        filename = QFileDialog.getOpenFileName(filter ='*.tsv')
        path = filename[0]
        with open(path, 'r') as f:
            data = f.read()
        lst_ = [[j for j in i.split('\t')] for i in data.split('\n')]
        i = 0
        while lst_[i][0] != 'TRAJECTORY_TYPES':
            i += 1
        j = 0
        while lst_[j][0] != 'MARKER_NAMES':
            j += 1
        self.market_name = lst_[j][1:]
        self.data = pd.DataFrame(lst_[i + 2:], columns=lst_[i + 1][:-1]).dropna()
        work_col = [i for i in self.data.columns if i.find('Type') == -1]
        self.data = self.data[work_col].astype(float)
        self.horizontalSlider.setMaximum(self.data.shape[0])
        #Включение графиков
        self.showMaximized()
        self.showNormal()
        self.calc()
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
        self.show_table()
        self.plot()

    def plot(self):
        data_dynamic = self.data  # следующий индекс - номер маркера
        t = self.time_index
        lst_x = [i for i in data_dynamic.columns if i.split()[0] in self.market_name and i.split()[1] == 'X']
        lst_z = [i for i in data_dynamic.columns if i.split()[0] in self.market_name and i.split()[1] == 'Z']
        x = data_dynamic[lst_x].values[t]
        y = data_dynamic[lst_z].values[t]
        self.sagital_widget.clear()
        self.sagital_widget.setBackground('w')
        self.sagital_widget.plot(x, y, pen=None, symbol='o', symbolSize=7)
        self.sagital_widget.showGrid(x=True, y=True)
        lst_y = [i for i in data_dynamic.columns if i.split()[0] in self.market_name and i.split()[1] == 'Y']
        lst_z = [i for i in data_dynamic.columns if i.split()[0] in self.market_name and i.split()[1] == 'Z']
        x = data_dynamic[lst_y].values[t]
        y = data_dynamic[lst_z].values[t]
        self.front_widget.clear()
        self.front_widget.setBackground('w')
        self.front_widget.plot(x, y, pen=None, symbol='o', symbolSize=7)
        self.front_widget.showGrid(x=True, y=True)

    def changeValue(self, value):  # Toolbar
        self.time_index = value

    def take_data_by_name(self,name):
        work_col = [i for i in self.data.columns if  i.split()[0] in name]
        return np.array([self.data[work_col].take([0], axis=1),self.data[work_col].take([1], axis=1),self.data[work_col].take([2], axis=1)])

    def calc(self):
        self.v2 = (self.take_data_by_name('L.PSIS') + self.take_data_by_name('R.PSIS') + self.take_data_by_name('L.ASIS') + self.take_data_by_name('R.ASIS')) / 4

        param_1 = self.param_1()
        param_2 = self.param_2()
        param_3 = self.param_3()
        param_4 = self.param_4()
        param_5 = self.param_5()
        param_6 = self.param_6()
        param_7 = self.param_7()
        param_8 = self.param_8()
        param_9 = self.param_9()
        param_10 = self.param_10()
        param_11 = self.param_11()
        param_12 = self.param_12()

        APA = self.APA()
        Dyn_Cobb = self.Dyn_Cobb()
        Dyn_LL = self.Dyn_LL()
        Dyn_PT = self.Dyn_PT()
        Dyn_SL = self.Dyn_SL()
        Dyn_SL_rotation = self.Dyn_SL_rotation()
        Dyn_TK = self.Dyn_TK()


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


    def param_9(self, ):
        # Сагиттальный наклон грудной клетки относительно пола
        param_1 = []
        vect_1 = self.take_data_by_name('T1') - self.take_data_by_name('S1')
        vect = np.array([1, 0, 0])
        for i in range(self.data.shape[0]):
            param_1.append(np.dot(vect, vect_1[:, i])[0])
        return param_1

    def param_10(self, ):
        # фронтальный наклон грудной клетки относительно пола
        vect_1 =self.take_data_by_name('T1') - self.take_data_by_name('S1')
        vect = np.array([0, 1, 0])
        param_2 = []
        for i in range(self.data.shape[0]):
            param_2.append(np.dot(vect, vect_1[:, i])[0])
        return param_2

    def param_11(self, ):
        # Сагиттальный наклон грудной клетки относительно таза
        vect = [1, 0, 0]
        vect_1 = self.take_data_by_name('T1') - self.v2
        param_3 = []
        for i in range(self.data.shape[0]):
            param_3.append(np.dot(vect, vect_1[:, i])[0])
        return param_3
    def param_12(self,):
        # фронтальный наклон грудной клетки относительно таза
        vect = [0, 1, 0]
        vect_1 = self.take_data_by_name('T1') - self.v2
        param_4 = []
        for i in range(self.data.shape[0]):
            param_4.append(np.dot(vect, vect_1[:, i])[0])
        return param_4
    def param_1(self, ):
        #'Вариант 1. сагиттальная плоскость'
        param_1 = []
        for i in range(self.data.shape[0]):
            param_1.append(np.linalg.norm(self.take_data_by_name('S1')[0:3:2,i] - self.take_data_by_name('T3')[0:3:2,i]))
        return param_1

    def param_2(self ):
        #  Вариант 1. фронтальная плоскость
        param_2 = []
        for i in range(self.data.shape[0]):

            param_2.append(np.linalg.norm(self.take_data_by_name('S1')[1:3,i] - self.take_data_by_name('T1')[1:3,i]))
        return param_2

    def param_3(self,):
        #         'Вариант 2. сагиттальная плоскость',
        param_3 = []
        for i in range(self.data.shape[0]):
            param_3.append(np.linalg.norm(self.take_data_by_name('T1')[0:3:2,i] - self.v2[0:3:2,i]))
        return param_3
    def param_4(self):
        #'Вариант 2. фронтальная плоскость',
        param_4 = []
        for i in range(self.data.shape[0]):
            param_4.append(np.linalg.norm(self.take_data_by_name('T1')[1:3,i] - self.v2[1:3,i]))
        return param_4
    def param_5(self,):
        # торсия груди относительно пола
        vect = [1, 0, 0]
        vect_1 = self.take_data_by_name('ES') - self.take_data_by_name('S1')
        param_5 = []
        for i in range(self.data.shape[0]):
            param_5.append(np.dot(vect, vect_1[:, i])[0])
        return param_5
    def param_6(self,):
        # торсия груди относительно таза
        vect = [1, 0, 0]
        vect_1 = self.take_data_by_name('ES') - self.v2
        param_6 = []
        for i in range(self.data.shape[0]):
            param_6.append(np.dot(vect, vect_1[:, i])[0])
        return param_6
    def param_7(self,):
        # наклон головы относительно пола
        vect = [1, 0, 0]
        vect_1 = self.take_data_by_name('ES') - self.take_data_by_name('S1')
        param_7 = []
        for i in range(self.data.shape[0]):
            param_7.append(np.dot(vect, vect_1[:, i])[0])
        return param_7

    def param_8(self,):
        # наклон головы  относительно тела
        vect = [1, 0, 0]
        vect_1 = self.take_data_by_name('T1') - self.v2
        param_8 = []
        for i in range(self.data.shape[0]):
            param_8.append(np.dot(vect, vect_1[:, i])[0])
        return param_8
    def Dyn_Cobb(self,):
        # calculation Dyn-Cobb angle
        # calculation Dyn-Cobb angle: angle between T9-T11 p39-c in plane ZY
        l1 = np.reshape(np.array(
            (self.take_data_by_name('c')[1] - self.take_data_by_name('P39')[1], self.take_data_by_name('c')[2] - self.take_data_by_name('P39')[2])),(2,3000))  # vector p39-c
        l2 = np.reshape(np.array(
            (self.take_data_by_name('T11')[1] - self.take_data_by_name('T9')[1], self.take_data_by_name('T11')[2] - self.take_data_by_name('T9')[2])),(2,3000))  # vector T9-T11

        len_l1 = []
        len_l2 = []
        Dyn_Cobb = []
        for i in range(len(l1[0])):
            len_l1.append(sqrt(l1[0][i] ** 2 + l1[1][i] ** 2))
            len_l2.append(sqrt(l2[0][i] ** 2 + l2[1][i] ** 2))
            Dyn_Cobb.append(acos(np.dot(l1[:, i], l2[:, i]) / (len_l1[i] * len_l2[i])))
        return Dyn_Cobb

    def Dyn_SL(self):
        # calculation Dyn-SL inclination : angle between R.ACR-L.ACR line Dyn-CVA(T1-S1) in plane ZY
        l1 = np.reshape(np.array(
            (self.take_data_by_name('R.ACR')[1] - self.take_data_by_name('L.ACR')[1], self.take_data_by_name('R.ACR')[2] - self.take_data_by_name('L.ACR')[2])),(2,3000))  # vector R.ACR-L.ACR
        l2 = np.reshape(np.array(
            (self.take_data_by_name('T1')[1] - self.take_data_by_name('S1')[1], self.take_data_by_name('T1')[2] - self.take_data_by_name('S1')[2])),(2,3000))  # vector T1-S1

        len_l1 = []
        len_l2 = []
        Dyn_SL = []
        for i in range(len(l1[0])):
            len_l1.append(sqrt(l1[0][i] ** 2 + l1[1][i] ** 2))
            len_l2.append(sqrt(l2[0][i] ** 2 + l2[1][i] ** 2))
            Dyn_SL.append(acos(np.dot(l1[:, i], l2[:, i]) / (len_l1[i] * len_l2[i])))
        return Dyn_SL

    def Dyn_TK(self):
        # Dyn-TK angle: angle between T9-T11 p39-c in plane XZ
        l1 = np.reshape(np.array(
            (self.take_data_by_name('P39')[0] - self.take_data_by_name('c')[0], self.take_data_by_name('P39')[2] - self.take_data_by_name('c')[2])),(2,3000))   # vector p39-c
        l2 = np.reshape(np.array(
            (self.take_data_by_name('T9')[0] - self.take_data_by_name('T11')[0], self.take_data_by_name('T9')[2] - self.take_data_by_name('T11')[2])),(2,3000))    # vector T9-T11
        len_l1 = []
        len_l2 = []
        Dyn_TK = []
        for i in range(len(l1[0])):
            len_l1.append(sqrt(l1[0][i] ** 2 + l1[1][i] ** 2))
            len_l2.append(sqrt(l2[0][i] ** 2 + l2[1][i] ** 2))
            Dyn_TK.append(acos(np.dot(l1[:, i], l2[:, i]) / (len_l1[i] * len_l2[i])))
        return Dyn_TK
    def Dyn_LL(self):
        # calculation Dyn-LL angle:   angle between T11-p114 L4-p114 in plane ZX
        l1 = np.reshape(np.array(
            (self.take_data_by_name('L4')[0] - self.take_data_by_name('p114')[0], self.take_data_by_name('L4')[2] - self.take_data_by_name('p114')[2])),(2,3000))  # vector L4-p114
        l2 = np.reshape(np.array(
            (self.take_data_by_name('T11')[0] - self.take_data_by_name('p114')[0], self.take_data_by_name('T11')[2] - self.take_data_by_name('p114')[2])),(2,3000))  # vector T11-p114


        len_l1 = []
        len_l2 = []
        Dyn_LL = []
        for i in range(len(l1[0])):
            len_l1.append(sqrt(l1[0][i] ** 2 + l1[1][i] ** 2))
            len_l2.append(sqrt(l2[0][i] ** 2 + l2[1][i] ** 2))
            Dyn_LL.append(acos(np.dot(l1[:, i], l2[:, i]) / (len_l1[i] * len_l2[i])))
        return Dyn_LL

    def Dyn_PT(self):
        # calculation  Dyn-PT two ways
        # calculation Dyn-PT : angle between L.PSIS-L.ASIS and normal vector к ZY in plane ZX
        l1 = np.reshape(np.array(
            (self.take_data_by_name('L.PSIS')[0] - self.take_data_by_name('L.ASIS')[0], self.take_data_by_name(' L.PSIS')[2] - self.take_data_by_name('L.ASIS')[2])),(2,3000)) # vector L.PSIS-L.ASIS
        l2 = []
        for i in range(len(l1[0])):
            l2.append(np.array([0, -1]))  # normal vector
        len_l1 = []
        len_l2 = []

        Dyn_PT_1 = []
        for i in range(len(l1[0])):
            len_l1.append(sqrt(l1[0][i] ** 2 + l1[1][i] ** 2))
            len_l2.append(sqrt(l2[i][0] ** 2 + l2[i][1] ** 2))
            Dyn_PT_1.append(acos(np.dot(l1[:, i], l2[ i]) / (len_l1[i] * len_l2[i])))
        # calculation Dyn-PT : angle between R.PSIS-R.ASIS and normal vector к ZY in plane ZX
        l1 = np.reshape(np.array(
            (self.take_data_by_name('R.PSIS')[0] - self.take_data_by_name('R.ASIS')[0], self.take_data_by_name('R.PSIS')[2] - self.take_data_by_name('R.ASIS')[2])),(2,3000)) # vector R.PSIS-R.ASIS
        l2 = []
        for i in range(len(l1[0])):
            l2.append(np.array([0, -1]))  # normal vector

        len_l1 = []
        len_l2 = []

        Dyn_PT_2 = []
        for i in range(len(l1[0])):
            len_l1.append(sqrt(l1[0][i] ** 2 + l1[1][i] ** 2))
            len_l2.append(sqrt(l2[i][0] ** 2 + l2[i][1] ** 2))
            Dyn_PT_2.append(acos(np.dot(l1[:, i], l2[i]) / (len_l1[i] * len_l2[i])))
        return Dyn_PT_2

    def Dyn_SL_rotation(self):

        # calculation Dyn_SL_rotation : angle between R.ACR-L.ACR and normal vector of ZX in plane XY\

        l1 = np.reshape(np.array(
            (self.take_data_by_name('R.ACR')[0] - self.take_data_by_name('L.ACR')[0],
             self.take_data_by_name('R.ACR')[2] - self.take_data_by_name('L.ACR')[2])), (2, 3000))  # vector R.ACR-L.ACR

        l2 = []
        for i in range(len(l1[0])):
            l2.append(np.array([0, -1]))  # normal vector

        len_l1 = []
        len_l2 = []

        Dyn_SL_rotation = []
        for i in range(len(l2[0])):
            len_l1.append(sqrt(l1[i][0] ** 2 + l1[i][1] ** 2))
            len_l2.append(sqrt(l2[i][0] ** 2 + l2[i][1] ** 2))
            Dyn_SL_rotation.append(acos(np.dot(l1[:, i], l2[i]) / (len_l1[i] * len_l2[i])))

        return Dyn_SL_rotation

    def pelvic_rotation(self):
        # calculation pelvic_rotation : angle between R.ASIS-L.ASIS  and normal vector of ZX in plane XY
        l1 = np.reshape(np.array(
            (self.take_data_by_name('R.ASIS')[0] - self.take_data_by_name('L.ASIS')[0],
             self.take_data_by_name('R.ASIS')[1] - self.take_data_by_name('L.ASIS')[1])),(2,3000))  # vector R.ASIS-L.ASIS

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

    def APA(self):
    # calculation APA : angle between R.ACR-L.ACR and R.ASIS-L.ASIS in plane XY
        l1 = np.reshape(np.array(
                (self.take_data_by_name('R.ACR')[0] - self.take_data_by_name('L.ACR')[0],
                 self.take_data_by_name('R.ACR')[1] - self.take_data_by_name('L.ACR')[1])),(2,3000))  # vector R.ACR-L.ACR
        l2 = np.reshape(np.array(
            (self.take_data_by_name('R.ASIS')[0] - self.take_data_by_name('L.ASIS')[0],
             self.take_data_by_name('R.ASIS')[1] - self.take_data_by_name('L.ASIS')[1])),(2,3000))  # vector R.ASIS-L.ASIS


        len_l1 = []
        len_l2 = []

        APA = []
        for i in range(len(l1[0])):
            len_l1.append(sqrt(l1[0][i] ** 2 + l1[1][i] ** 2))
            len_l2.append(sqrt(l2[0][i] ** 2 + l2[1][i] ** 2))
            APA.append(acos(np.dot(l1[:, i], l2[:, i]) / (len_l1[i] * len_l2[i])))
        return APA




    def show_table(self):

        model_2 = PandasModel(
            self.new_data[self.time_index].apply(lambda x: round(x, 8)).to_frame(name="time " + str(self.time_index)))
        model_1 = PandasModel(
            self.old_data[self.time_index].apply(lambda x: round(x, 8)).to_frame(name="time " + str(self.time_index)))
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

import sys
from functools import reduce
from math import sqrt, acos

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import scipy.io
import scipy.io
import scipy.signal
from PyQt5.QtWidgets import QFileDialog

matplotlib.use('Qt5Agg')
import os

from PyQt5 import QtWidgets

from PyQt5 import QtCore

import design_1_0_ as design

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
                           'Вариант 2. фронтальная плоскость']
        self.names_old = ['acromion pelvis angle', 'Cobb angle', 'lumbar lordosis', 'pelvic tilt', 'shoulder line inclination', 'shoulder line rotation', 'thoracic kyphosis']


        self.parameter_plot.setBackground('w')
        self.comboBox.addItems(self.names_news)
        self.comboBox.addItems(self.names_old)
        self.comboBox.currentTextChanged.connect(self.mini_plots)
        self.time_index = 1
        self.data_calc = None
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
        df = self.data_calc
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
        self.v2 = (self.take_data_by_name('L_IAS') + self.take_data_by_name('R_IAS') + self.take_data_by_name('L_IPS') + self.take_data_by_name('R_IPS')) / 4

        param_1 = self.param_1()
        param_2 = self.param_2()
        param_3 = self.param_3()
        param_4 = self.param_4()


        data_1 = pd.DataFrame([param_1, param_2, param_3, param_4],
                              index=self.names_news)

        self.data_calc = data_1

    def step_separator(marker, i_=2):

        for coordinate in marker:
            if 'Y' in coordinate:
                marker_y = coordinate
            if 'Z' in coordinate:
                marker_z = coordinate
            if 'X' in coordinate:
                marker_x = coordinate
        # Проверка ориентации
        if marker[marker_x][0] < marker[marker_x][len(marker[marker_x]) - 1]:
            marker[marker_y] = marker[marker_y] * (-1)
            marker[marker_z] = marker[marker_z] * (-1)
        # фильтрация частот
        #  в статье 2 HZ
        acs_AP = marker[marker_y].diff().diff().fillna(value=0).values
        b, a = scipy.signal.butter(11, i_, 'lowpass', fs=100)
        fil_acs_AP = scipy.signal.filtfilt(b, a, acs_AP)
        binary_acs_AP = [True if i > 0 else False for i in fil_acs_AP]
        vel_vertical = marker[marker_z].diff().fillna(value=0).values
        # Разбиения на промежутки по отфильтрованому сигналу Z
        selection = 4  # половина промежутка на котором ищу максимум всегда четный
        half_selection = int(selection / 2)
        point_HS = []
        point_TO = []
        # Момент соприкосновения
        HS_T = None
        T_ = False
        for i in range(half_selection, len(vel_vertical) - half_selection):
            min_left = reduce(lambda x, y: x and y,
                              [True if vel_vertical[i] < j else False for j in vel_vertical[i - half_selection:i]])
            min_right = reduce(lambda x, y: x and y,
                               [True if vel_vertical[i] < j else False for j in
                                vel_vertical[i + 1:i + half_selection + 1]])
            max_left = reduce(lambda x, y: x and y,
                              [True if vel_vertical[i] > j else False for j in vel_vertical[i - half_selection:i]])
            max_right = reduce(lambda x, y: x and y,
                               [True if vel_vertical[i] > j else False for j in
                                vel_vertical[i + 1:i + half_selection + 1]])
            if ((min_left and min_right) or (max_left and max_right)) and binary_acs_AP[i]:
                HS_T = i
                T_ = True
            if not binary_acs_AP[i] and T_:
                point_HS.append(HS_T)
                T_ = False
        # момент отрыва
        T_ = True
        for i in range(half_selection, len(vel_vertical) - half_selection):
            min_left = reduce(lambda x, y: x and y,
                              [True if vel_vertical[i] < j else False for j in vel_vertical[i - half_selection:i]])
            min_right = reduce(lambda x, y: x and y,
                               [True if vel_vertical[i] < j else False for j in
                                vel_vertical[i + 1:i + half_selection + 1]])
            max_left = reduce(lambda x, y: x and y,
                              [True if vel_vertical[i] > j else False for j in vel_vertical[i - half_selection:i]])
            max_right = reduce(lambda x, y: x and y,
                               [True if vel_vertical[i] > j else False for j in
                                vel_vertical[i + 1:i + half_selection + 1]])

            if ((max_left and max_right) or (min_left and min_right)) and binary_acs_AP[i] and T_ == True:
                point_TO.append(i)
                T_ = False
            if not binary_acs_AP[i]:
                T_ = True
        return [point_HS, point_TO]



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





    def show_table(self):


        model_1 = PandasModel(
            self.data_calc[self.time_index].apply(lambda x: round(x, 8)).to_frame(name="time " + str(self.time_index)))
        self.tableView.setModel(model_1)
        # переделать вывод использовать тул бар для вывода отрисовать графики


def main():
    app = QtWidgets.QApplication(sys.argv)  # Новый экземпляр QApplication
    window = ExampleApp()  # Создаём объект класса ExampleApp
    window.show()  # Показываем окно
    app.exec_()  # и запускаем приложение


if __name__ == '__main__':  # Если мы запускаем файл напрямую, а не импортируем
    main()  # то запускаем функцию main()

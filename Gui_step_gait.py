from math import sqrt, acos
import scipy.io
import numpy as np
import pandas as pd
from PyQt5 import uic

import matplotlib
import matplotlib.pyplot as plt
import pyqtgraph
import sys
from functools import reduce
from sys import argv
import scipy
import scipy.signal
import matplotlib.pyplot as plt
import pandas as pd
from PyQt5.QtCore import QFile, Qt
from PyQt5.QtWidgets import QFileDialog

matplotlib.use('Qt5Agg')
import os

from PyQt5 import QtWidgets

from PyQt5 import QtCore

import step_gui as design
from matplotlib.pyplot import figure
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
import pyqtgraph as pg
from pyqtgraph import PlotWidget

def step_separator(marker, i_=2,HS = None, TO = None):

    for coordinate in marker:
        if 'Y' in coordinate:
            marker_y = coordinate
        if 'Z' in coordinate:
            marker_z = coordinate
        if 'X' in coordinate:
            marker_x = coordinate

    # фильтрация частот
    #  в статье 2 HZ
    acs_AP = marker[marker_y].diff().diff().fillna(value=0).values
    b, a = scipy.signal.butter(11, i_, 'lowpass', fs=100)
    fil_acs_AP = scipy.signal.filtfilt(b, a, acs_AP)
    binary_acs_AP = [True if i > 0 else False for i in fil_acs_AP]
    vel_vertical = marker[marker_z].diff().fillna(value=0).values
    # Разбиения на промежутки по отфильтрованому сигналу Z
    selection = 4#половина промежутка на котором ищу максимум всегда четный
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
                           [True if vel_vertical[i] < j else False for j in vel_vertical[i + 1:i + half_selection + 1]])
        max_left = reduce(lambda x, y: x and y,
                          [True if vel_vertical[i] > j else False for j in vel_vertical[i - half_selection:i]])
        max_right = reduce(lambda x, y: x and y,
                           [True if vel_vertical[i] > j else False for j in vel_vertical[i + 1:i + half_selection + 1]])
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
                           [True if vel_vertical[i] < j else False for j in vel_vertical[i + 1:i + half_selection + 1]])
        max_left = reduce(lambda x, y: x and y,
                          [True if vel_vertical[i] > j else False for j in vel_vertical[i - half_selection:i]])
        max_right = reduce(lambda x, y: x and y,
                           [True if vel_vertical[i] > j else False for j in vel_vertical[i + 1:i + half_selection + 1]])

        if ((max_left and max_right) or (min_left and min_right)) and binary_acs_AP[i] and T_ == True:
            point_TO.append(i)
            T_ = False
        if not binary_acs_AP[i]:
            T_ = True
    return [point_HS, point_TO]

class ExampleApp(QtWidgets.QMainWindow,design.Ui_MainWindow):
    def __init__(self):
        # Это здесь нужно для доступа к переменным, методам
        # и т.д. в файле design.py
        super().__init__()
        self.setupUi(self)

         # загрузка
        self.load_Button.clicked.connect(self.download_file)
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
        k = 0
        Events = []
        while lst_[k][0] == 'EVENT' or len(Events) == 0:
            if lst_[k][0] == 'EVENT':
                Events.append(lst_[k])
            k += 1
        RHS = [int(i[2]) for i in Events if 'RHS' in i]
        LHS = [int(i[2]) for i in Events if 'LHS' in i]
        RTO = [int(i[2]) for i in Events if 'RTO' in i]
        LTO = [int(i[2]) for i in Events if 'LTO' in i]
        self.market_name = lst_[j][1:]
        self.data = pd.DataFrame(lst_[i + 2:], columns=lst_[i + 1][:-1]).dropna()
        work_col = [i for i in self.data.columns if i.find('Type') == -1]
        self.data = self.data[work_col].astype(float)
        times = self.data['Time'].values
        point_HS, point_TO = step_separator(self.data[['R_SAE Y', 'R_SAE Z']])
        point_HS = [times[i] for i in point_HS]
        point_TO = [times[i] for i in point_TO]
        ## ГРАФИКИ
        figure(figsize=(10, 10), dpi=80)
        plt.scatter(point_HS, [1 for i in (point_HS)], color='red', s=36, label='manual RHS')
        plt.scatter(point_TO, [0 for i in (point_TO)], color='green', s=36, label='manual RTO')

        plt.scatter(self.data[[i in RHS for i in self.data['Frame']]]['Time'].to_list(), [1 for i in (RHS)], color='black', s=30,
                    label='marking RHS')
        plt.scatter(self.data[[i in RTO for i in self.data['Frame']]]['Time'].to_list(), [0 for i in (RTO)], color='purple', s=30,
                    label='marking RTO')
        plt.plot(times, self.data['R_SAE Y'].diff().fillna(value=0).values, label='velocity R_SAE Y')
        plt.legend()
        plt.title("plot of algorithm")
        plt.margins(0, .05)
        plt.show()



def main():
    app = QtWidgets.QApplication(sys.argv)  # Новый экземпляр QApplication
    window = ExampleApp()  # Создаём объект класса ExampleApp
    window.show()  # Показываем окно
    app.exec_()  # и запускаем приложение


if __name__ == '__main__':  # Если мы запускаем файл напрямую, а не импортируем
    main()  # то запускаем функцию main()

# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'data/design_1_2_2.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from pyqtgraph import PlotWidget

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1315, 860)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(10, 30, 1301, 751))
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.frame = QtWidgets.QFrame(self.tab)
        self.frame.setGeometry(QtCore.QRect(0, 0, 1301, 721))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.tableView = QtWidgets.QTableView(self.frame)
        self.tableView.setGeometry(QtCore.QRect(0, 440, 471, 281))
        self.tableView.setObjectName("tableView")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.frame)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(860, 450, 431, 271))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.parameter_plot = PlotWidget(self.verticalLayoutWidget)
        self.parameter_plot.setObjectName("parameter_plot")
        self.verticalLayout_2.addWidget(self.parameter_plot)
        self.comboBox = QtWidgets.QComboBox(self.frame)
        self.comboBox.setGeometry(QtCore.QRect(860, 420, 431, 22))
        self.comboBox.setObjectName("comboBox")
        self.layoutWidget = QtWidgets.QWidget(self.frame)
        self.layoutWidget.setGeometry(QtCore.QRect(1000, 289, 295, 111))
        self.layoutWidget.setObjectName("layoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.layoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.horizontalSlider = QtWidgets.QSlider(self.layoutWidget)
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider.setObjectName("horizontalSlider")
        self.gridLayout.addWidget(self.horizontalSlider, 1, 0, 1, 3)
        self.lcdNumber = QtWidgets.QLCDNumber(self.layoutWidget)
        self.lcdNumber.setObjectName("lcdNumber")
        self.gridLayout.addWidget(self.lcdNumber, 0, 2, 1, 1)
        self.pushButton = QtWidgets.QPushButton(self.layoutWidget)
        self.pushButton.setObjectName("pushButton")
        self.gridLayout.addWidget(self.pushButton, 3, 0, 1, 1)
        self.label = QtWidgets.QLabel(self.layoutWidget)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 2)
        self.playButton = QtWidgets.QPushButton(self.layoutWidget)
        self.playButton.setObjectName("playButton")
        self.gridLayout.addWidget(self.playButton, 3, 1, 1, 1)
        self.stopButton = QtWidgets.QPushButton(self.layoutWidget)
        self.stopButton.setObjectName("stopButton")
        self.gridLayout.addWidget(self.stopButton, 3, 2, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.frame)
        self.label_3.setGeometry(QtCore.QRect(0, 25, 151, 21))
        self.label_3.setObjectName("label_3")
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.frame)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(0, 50, 951, 311))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.sagital_widget = PlotWidget(self.horizontalLayoutWidget)
        self.sagital_widget.setObjectName("sagital_widget")
        self.horizontalLayout.addWidget(self.sagital_widget)
        self.front_widget = PlotWidget(self.horizontalLayoutWidget)
        self.front_widget.setObjectName("front_widget")
        self.horizontalLayout.addWidget(self.front_widget)
        self.label_2 = QtWidgets.QLabel(self.frame)
        self.label_2.setGeometry(QtCore.QRect(480, 30, 161, 16))
        self.label_2.setObjectName("label_2")
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.frame_2 = QtWidgets.QFrame(self.tab_2)
        self.frame_2.setGeometry(QtCore.QRect(0, 0, 1291, 721))
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.verticalLayoutWidget_2 = QtWidgets.QWidget(self.frame_2)
        self.verticalLayoutWidget_2.setGeometry(QtCore.QRect(270, 20, 941, 221))
        self.verticalLayoutWidget_2.setObjectName("verticalLayoutWidget_2")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_2)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.mean_plot_s = PlotWidget(self.verticalLayoutWidget_2)
        self.mean_plot_s.setObjectName("mean_plot_s")
        self.verticalLayout_3.addWidget(self.mean_plot_s)
        self.verticalLayoutWidget_4 = QtWidgets.QWidget(self.frame_2)
        self.verticalLayoutWidget_4.setGeometry(QtCore.QRect(269, 260, 941, 361))
        self.verticalLayoutWidget_4.setObjectName("verticalLayoutWidget_4")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_4)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.planeBox_s = QtWidgets.QComboBox(self.verticalLayoutWidget_4)
        self.planeBox_s.setObjectName("planeBox_s")
        self.verticalLayout.addWidget(self.planeBox_s)
        self.tableView_s = QtWidgets.QTableView(self.verticalLayoutWidget_4)
        self.tableView_s.setObjectName("tableView_s")
        self.verticalLayout.addWidget(self.tableView_s)
        self.tabWidget.addTab(self.tab_2, "")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.frame_3 = QtWidgets.QFrame(self.tab_3)
        self.frame_3.setGeometry(QtCore.QRect(0, 0, 1301, 721))
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.verticalLayoutWidget_5 = QtWidgets.QWidget(self.frame_3)
        self.verticalLayoutWidget_5.setGeometry(QtCore.QRect(269, 260, 941, 361))
        self.verticalLayoutWidget_5.setObjectName("verticalLayoutWidget_5")
        self.verticalLayout_8 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_5)
        self.verticalLayout_8.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.planeBox_f = QtWidgets.QComboBox(self.verticalLayoutWidget_5)
        self.planeBox_f.setObjectName("planeBox_f")
        self.verticalLayout_8.addWidget(self.planeBox_f)
        self.tableView_f = QtWidgets.QTableView(self.verticalLayoutWidget_5)
        self.tableView_f.setObjectName("tableView_f")
        self.verticalLayout_8.addWidget(self.tableView_f)
        self.verticalLayoutWidget_3 = QtWidgets.QWidget(self.frame_3)
        self.verticalLayoutWidget_3.setGeometry(QtCore.QRect(270, 20, 941, 221))
        self.verticalLayoutWidget_3.setObjectName("verticalLayoutWidget_3")
        self.verticalLayout_9 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_3)
        self.verticalLayout_9.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.mean_plot_f = PlotWidget(self.verticalLayoutWidget_3)
        self.mean_plot_f.setObjectName("mean_plot_f")
        self.verticalLayout_9.addWidget(self.mean_plot_f)
        self.tabWidget.addTab(self.tab_3, "")
        self.tab_4 = QtWidgets.QWidget()
        self.tab_4.setObjectName("tab_4")
        self.frame_4 = QtWidgets.QFrame(self.tab_4)
        self.frame_4.setGeometry(QtCore.QRect(0, 0, 1301, 731))
        self.frame_4.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_4.setObjectName("frame_4")
        self.verticalLayoutWidget_6 = QtWidgets.QWidget(self.frame_4)
        self.verticalLayoutWidget_6.setGeometry(QtCore.QRect(270, 20, 941, 221))
        self.verticalLayoutWidget_6.setObjectName("verticalLayoutWidget_6")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_6)
        self.verticalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.mean_plot_a = PlotWidget(self.verticalLayoutWidget_6)
        self.mean_plot_a.setObjectName("mean_plot_a")
        self.verticalLayout_6.addWidget(self.mean_plot_a)
        self.verticalLayoutWidget_7 = QtWidgets.QWidget(self.frame_4)
        self.verticalLayoutWidget_7.setGeometry(QtCore.QRect(270, 260, 941, 361))
        self.verticalLayoutWidget_7.setObjectName("verticalLayoutWidget_7")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_7)
        self.verticalLayout_7.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.planeBox_a = QtWidgets.QComboBox(self.verticalLayoutWidget_7)
        self.planeBox_a.setObjectName("planeBox_a")
        self.verticalLayout_7.addWidget(self.planeBox_a)
        self.tableView_a = QtWidgets.QTableView(self.verticalLayoutWidget_7)
        self.tableView_a.setObjectName("tableView_a")
        self.verticalLayout_7.addWidget(self.tableView_a)
        self.tabWidget.addTab(self.tab_4, "")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(1160, 780, 151, 21))
        self.pushButton_2.setObjectName("pushButton_2")
        self.fileBox = QtWidgets.QComboBox(self.centralwidget)
        self.fileBox.setGeometry(QtCore.QRect(880, 20, 431, 22))
        self.fileBox.setObjectName("fileBox")
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(720, 20, 131, 20))
        self.label_8.setObjectName("label_8")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1315, 26))
        self.menubar.setObjectName("menubar")
        self.menu = QtWidgets.QMenu(self.menubar)
        self.menu.setObjectName("menu")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.action = QtWidgets.QAction(MainWindow)
        self.action.setObjectName("action")
        self.action_2 = QtWidgets.QAction(MainWindow)
        self.action_2.setObjectName("action_2")
        self.action_3 = QtWidgets.QAction(MainWindow)
        self.action_3.setObjectName("action_3")
        self.action_m = QtWidgets.QAction(MainWindow)
        self.action_m.setObjectName("action_m")
        self.action_p = QtWidgets.QAction(MainWindow)
        self.action_p.setObjectName("action_p")
        self.action_l = QtWidgets.QAction(MainWindow)
        self.action_l.setObjectName("action_l")
        self.menu.addAction(self.action_m)
        self.menu.addAction(self.action_p)
        self.menu.addAction(self.action_l)
        self.menubar.addAction(self.menu.menuAction())

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "GUI"))
        self.pushButton.setText(_translate("MainWindow", "перейти"))
        self.label.setText(_translate("MainWindow", "временной промежуток:"))
        self.playButton.setText(_translate("MainWindow", "старт"))
        self.stopButton.setText(_translate("MainWindow", "стоп"))
        self.label_3.setText(_translate("MainWindow", "сагиттальная плоскость"))
        self.label_2.setText(_translate("MainWindow", "фронтальная плоскость"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "основное окно"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "саггитальная плоскость"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), _translate("MainWindow", "фронтальная плоскость"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_4), _translate("MainWindow", "аксиальная плоскость"))
        self.pushButton_2.setText(_translate("MainWindow", "выгрузить"))
        self.label_8.setText(_translate("MainWindow", "Выбор пациента"))
        self.menu.setTitle(_translate("MainWindow", "меню"))
        self.action.setText(_translate("MainWindow", "основное окно"))
        self.action_2.setText(_translate("MainWindow", "фронтальная плоскость"))
        self.action_3.setText(_translate("MainWindow", "сагитальная плоскость"))
        self.action_m.setText(_translate("MainWindow", "переменовать маркер"))
        self.action_p.setText(_translate("MainWindow", "добавить параметр"))
        self.action_l.setText(_translate("MainWindow", "загрузить файл"))
from pyqtgraph import PlotWidget

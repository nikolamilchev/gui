import scipy.io
import threading
import numpy as np
import pandas as pd
from math import sqrt, acos

def calc():
    walk_5km0001 = scipy.io.loadmat('walk 5km0001.mat')
    statica0001  =  scipy.io.loadmat('statica0001.mat')
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

    # фронтальный наклон грудной клетки относительно таза
    vect = [0, 1, 0]
    vect_1 = data_dynamic[5, 0:3] - v2
    param_4 = []
    for i in range(3000):
        param_4.append(np.dot(vect, vect_1[:, i]))

    # торсия груди относительно пола
    vect = [1, 0, 0]
    vect_1 = data_dynamic[15, 0:3] - data_dynamic[10, 0:3]
    param_5 = []
    for i in range(3000):
        param_5.append(np.dot(vect, vect_1[:, i]))

    # торсия груди относительно таза
    vect = [1, 0, 0]
    vect_1 = data_dynamic[15, 0:3] - v2
    param_6 = []
    for i in range(3000):
        param_6.append(np.dot(vect, vect_1[:, i]))
    # наклон головы относительно пола
    vect = [1, 0, 0]
    vect_1 = data_dynamic[2, 0:3] - data_dynamic[10, 0:3]
    param_7 = []
    for i in range(3000):
        param_7.append(np.dot(vect, vect_1[:, i]))

    # наклон головы  относительно тела
    vect = [1, 0, 0]
    vect_1 = data_dynamic[2, 0:3] - v2
    param_8 = []
    for i in range(3000):
        param_8.append(np.dot(vect, vect_1[:, i]))

    # calculation Dyn-Cobb angle
    # calculation Dyn-Cobb angle: angle between T9-T11 p39-c in plane ZY
    l1 = np.array((data_dynamic[5, 1] - data_dynamic[4, 1], data_dynamic[5, 2] - data_dynamic[4, 2]))  # vector p39-c
    l2 = np.array((data_dynamic[7, 1] - data_dynamic[6, 1], data_dynamic[7, 2] - data_dynamic[6, 2]))  # vector T9-T11

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
    l2 = np.array((data_dynamic[2, 1] - data_dynamic[10, 1], data_dynamic[2, 2] - data_dynamic[10, 2]))  # vector T1-S1

    len_l1 = []
    len_l2 = []
    Dyn_SL = []
    for i in range(len(l1[0])):
        len_l1.append(sqrt(l1[0][i] ** 2 + l1[1][i] ** 2))
        len_l2.append(sqrt(l2[0][i] ** 2 + l2[1][i] ** 2))
        Dyn_SL.append(acos(np.dot(l1[:, i], l2[:, i]) / (len_l1[i] * len_l2[i])))

    # Dyn-TK angle: angle between T9-T11 p39-c in plane XZ
    l1 = np.array((data_dynamic[4, 0] - data_dynamic[5, 0], data_dynamic[4, 2] - data_dynamic[5, 2]))  # vector p39-c
    l2 = np.array((data_dynamic[6, 0] - data_dynamic[7, 0], data_dynamic[6, 2] - data_dynamic[7, 2]))  # vector T9-T11

    len_l1 = []
    len_l2 = []
    Dyn_TK = []
    for i in range(len(l1[0])):
        len_l1.append(sqrt(l1[0][i] ** 2 + l1[1][i] ** 2))
        len_l2.append(sqrt(l2[0][i] ** 2 + l2[1][i] ** 2))
        Dyn_TK.append(acos(np.dot(l1[:, i], l2[:, i]) / (len_l1[i] * len_l2[i])))

    # calculation Dyn-LL angle:   angle between T11-p114 L4-p114 in plane ZX
    l1 = np.array((data_dynamic[9, 0] - data_dynamic[8, 0], data_dynamic[9, 2] - data_dynamic[8, 2]))  # vector L4-p114
    l2 = np.array((data_dynamic[8, 0] - data_dynamic[7, 0], data_dynamic[8, 2] - data_dynamic[9, 2]))  # vector T11-p114

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
        (data_dynamic[11, 0] - data_dynamic[13, 1], data_dynamic[11, 2] - data_dynamic[13, 2]))  # vector L.PSIS-L.ASIS
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
        (data_dynamic[12, 0] - data_dynamic[14, 0], data_dynamic[12, 2] - data_dynamic[14, 2]))  # vector L.PSIS-L.ASIS
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
        (data_dynamic[12, 0] - data_dynamic[13, 0], data_dynamic[12, 2] - data_dynamic[13, 2]))  # vector R.ASIS-L.ASIS

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
        (data_dynamic[13, 0] - data_dynamic[14, 0], data_dynamic[13, 2] - data_dynamic[14, 2]))  # vector R.ASIS-L.ASIS

    len_l1 = []
    len_l2 = []

    APA = []
    for i in range(len(l1[0])):
        len_l1.append(sqrt(l1[0][i] ** 2 + l1[1][i] ** 2))
        len_l2.append(sqrt(l2[0][i] ** 2 + l2[1][i] ** 2))
        APA.append(acos(np.dot(l1[:, i], l2[:, i]) / (len_l1[i] * len_l2[i])))
    data_1 = pd.DataFrame([param_1, param_2, param_3, param_4, param_5, param_6, param_7, param_8],
                          index=['Сагиттальный наклон грудной клетки относительно пола ',
                                 'фронтальный наклон грудной клетки относительно пола',
                                 'Сагиттальный наклон грудной клетки относительно таза',
                                 'фронтальный наклон грудной клетки относительно таза',
                                 'торсия груди относительно пола', 'торсия груди относительно таза',
                                 'наклон головы относительно пола ', 'наклон головы  относительно тела'])
    data_2 = pd.DataFrame([APA, Dyn_Cobb, Dyn_LL, Dyn_PT, Dyn_SL, Dyn_SL_rotation, Dyn_TK],
                          index=['APA', 'Dyn_Cobb', 'Dyn_LL', 'Dyn_PT', 'Dyn_SL', 'Dyn_SL_rotation', 'Dyn_TK'])
    print(data_1.shape)
    print(data_2.shape)
    data = pd.concat([data_1, data_2])

    return data

res = calc()
print(res)
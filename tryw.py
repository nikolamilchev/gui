def movement_d(x):
    return np.array([[data['Time'][0] * cos(data['Time'][0] * x[0] + x[2]), cos(data['Time'][0] * x[0] + x[2]),
                      data['Time'][0] * cos(data['Time'][0] * x[1] + x[3]), cos(data['Time'][0] * x[1] + x[3])],
                     [data['Time'][1] * cos(data['Time'][1] * x[0] + x[2]), cos(data['Time'][1] * x[0] + x[2]),
                      data['Time'][1] * cos(data['Time'][1] * x[1] + x[3]), cos(data['Time'][1] * x[1] + x[3])],
                     [data['Time'][2] * cos(data['Time'][2] * x[0] + x[2]), cos(data['Time'][2] * x[0] + x[2]),
                      data['Time'][2] * cos(data['Time'][2] * x[1] + x[3]), cos(data['Time'][2] * x[1] + x[3])],
                     [data['Time'][3] * cos(data['Time'][3] * x[0] + x[2]), cos(data['Time'][3] * x[0] + x[2]),
                      data['Time'][3] * cos(data['Time'][3] * x[1] + x[3]), cos(data['Time'][3] * x[1] + x[3])]])


def movement_(x):
    return [sin(data['Time'][0] * x[0] + x[2]) + sin(data['Time'][0] * x[1] + x[3]) -
            data['L_IPS Z'][0],
            sin(data['Time'][1] * x[0] + x[2]) + sin(data['Time'][1] * x[1] + x[3]) -
            data['L_IPS Z'][1],
            sin(data['Time'][2] * x[0] + x[2]) + sin(data['Time'][2] * x[1] + x[3]) -
            data['L_IPS Z'][2],
            sin(data['Time'][3] * x[0] + x[2]) + sin(data['Time'][3] * x[1] + x[3]) -
            data['L_IPS Z'][3]]

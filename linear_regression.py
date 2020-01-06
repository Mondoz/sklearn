import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model


def diagonal_compute(a,b,c):
    return math.sqrt(a**2 + b**2 + c**2)


model1_x_list = []
model1_y_list = []

model2_x_list = []
model2_y_list = []
with open('data/size_train.csv','r') as f:
    for lines in f.readlines():
        split = lines.split(',')
        label = split[1].strip()
        diameter = split[2].strip()
        if label == '4':
            model2_y_list.append([float(diameter)])
            diagonal = diagonal_compute(float(split[3]), float(split[4]), float(split[5]))
            model2_x_list.append([diagonal])
        else:
            model1_y_list.append([float(diameter)])
            diagonal = diagonal_compute(float(split[3]),float(split[4]),float(split[5]))
            model1_x_list.append([diagonal])

model1 = linear_model.LinearRegression()
model1.fit(model1_x_list,model1_y_list)

model2 = linear_model.LinearRegression()
model2.fit(model2_x_list,model2_y_list)

# plt.xlabel('diagonal')
# plt.ylabel('diameter')
# plt.xlim((0, 100))
# x_ticks = np.arange(0, 100, 10)
# plt.xticks(x_ticks)
# plt.plot(model1_x_list,model1_y_list,'k.')
# plt.plot(model2_x_list,model2_y_list,'k.')
# plt.show()

rf = open('data/size_result.csv','w')

with open('data/size_predict.csv','r') as f:
    for lines in f.readlines():
        split = lines.split(',')
        if len(split) < 3: continue
        label = split[1].strip()
        diameter = 0
        diagonal = diagonal_compute(float(split[2]), float(split[3]), float(split[4]))
        if label == '4':
            diameter = model2.predict([[diagonal]])[0][0]
        else:
            diameter = model1.predict([[diagonal]])[0][0]
        rf.write(lines.strip() + ',' + str(diameter) + '\n')

rf.close()
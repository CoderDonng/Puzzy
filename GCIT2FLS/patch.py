import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from CIT2.Gen_CIT2 import Gen_CIT2
from common.Input import Input
from common.Output import Output
from common.plot import PlotFun

if __name__ == '__main__':
    params = []
    with open("MFparams.txt", "r") as f:  # 打开文件
        data = f.readlines()  # 读取文件

    for i in iter(data):
        num = float(i)
        params.append(num)

    data = pd.read_csv("../data/diabetes.csv")
    p_data = data.loc[data['Outcome'] == 1, :].sample(250)
    n_data = data.loc[data['Outcome'] == 0, :].sample(250)
    data = pd.concat([p_data, n_data])
    X = data.drop(columns='Outcome')
    Y = data["Outcome"]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test)

    ### support
    supports = []
    for i in range(X_train.shape[1]):
        supports.append([X_train.min(axis=0)[i], X_train.max(axis=0)[i]])
    supports.append([0, 1])
    # 设置每个attribute和output的颗粒度
    Graininesses = [3, 3, 3, 3, 2]
    # 构造每个attr的输入类对象
    attr_Input = []
    for i in range(len(Graininesses) - 1):
        attr_Input.append(Input("attr_" + str(i + 1) + "_Input", supports[i]))
    # 输出类对象
    output = Output("Output", [0.0, 1.0])

    Attr_MFs = []
    for i in range(len(Graininesses) - 1):
        Attr_MFs.append([])
        Attr_MFs[i].append(Gen_CIT2("attr_" + str(i + 1) + "_level_1_mf", "T1MF_Triangular",
                                    [[params[i * 18 + 0], params[i * 18 + 1]],
                                     [params[i * 18 + 2], params[i * 18 + 3]],
                                     [params[i * 18 + 6], params[i * 18 + 7]]]))
        Attr_MFs[i].append(Gen_CIT2("attr_" + str(i + 1) + "_level_2_mf", "T1MF_Triangular",
                                    [[params[i * 18 + 4], params[i * 18 + 5]],
                                     [params[i * 18 + 8], params[i * 18 + 9]],
                                     [params[i * 18 + 12], params[i * 18 + 13]]]))
        Attr_MFs[i].append(Gen_CIT2("attr_" + str(i + 1) + "_level_3_mf", "T1MF_Triangular",
                                    [[params[i * 18 + 10], params[i * 18 + 11]],
                                     [params[i * 18 + 14], params[i * 18 + 15]],
                                     [params[i * 18 + 16], params[i * 18 + 17]]]))

    # 构建后件模糊数
    opt_MFs = []
    opt_MFs.append(Gen_CIT2("output_level_1", "T1MF_Triangular",
                            [[params[-12], params[-11]], [params[-10], params[-9]],
                             [params[-6], params[-5]]]))
    opt_MFs.append(Gen_CIT2("output_level_2", "T1MF_Triangular",
                            [[params[-8], params[-7]], [params[-4], params[-3]],
                             [params[-2], params[-1]]]))

    for m in range(len(Graininesses) - 1):
        plt.figure('the mfs of att['+str(m+1)+']')
        PlotFun().plot_IT2_MFs(Attr_MFs[m], 100000, attr_Input[m].getDomain(), [0, 1.0], False)
        plt.legend(['low', 'medium', 'high'])

    plt.figure('the mfs of output')
    PlotFun().plot_IT2_MFs(opt_MFs, 10000, output.getDomain(), [0, 1.0], False)
    plt.legend(['low probability', 'high probability'])
    plt.show()

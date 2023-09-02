from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm, neighbors
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from keras.models import Sequential
from keras.layers.core import Dense, Activation
import warnings
warnings.filterwarnings('ignore')


def logisticRegression(X_train, X_test, Y_train, Y_test):

    model = LogisticRegression()
    model.fit(X_train, Y_train)
    # print("LogisticRegression trainScore:" + str(model.score(X_train, Y_train))+" testScore:"+str(model.score(X_test,Y_test)))
    return model.score(X_train, Y_train), model.score(X_test,Y_test)


def Decisiontree(X_train, X_test, Y_train, Y_test):
    # trainScore = []
    # testScore = []

    # for depth in range(1, 10):
    #     model = DecisionTreeClassifier(max_depth=depth, random_state=0)
    #     model.fit(X_train, Y_train)
    #     trainScore.append(model.score(X_train, Y_train))
    #     testScore.append(model.score(X_test, Y_test))
    # plt.figure('decision tree')
    # plt.plot(trainScore, 'r.-')
    # plt.plot(testScore, 'b.-')
    # plt.legend(['train', 'test'])
    # plt.title('the score of decision tree')
    # plt.xlabel('depth')
    # plt.ylabel('accuracy')
    model = DecisionTreeClassifier(max_depth=4, random_state=0)
    model.fit(X_train, Y_train)
    return model.score(X_train, Y_train), model.score(X_test, Y_test)
    # print("Decision tree testScore:" + str(max(testScore)))

def Svm(X_train, X_test, Y_train, Y_test):
    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
    # trainScore = []
    # testScore = []
    #
    # for c in range(1, 6):
    #     for ga in range(1,3):
    #     # svcTest = svm.LinearSVC(C=c * 0.02, loss='hinge', max_iter=100)
    #     #     svc =  svm.SVC(C=c*0.05, gamma=ga, kernel='rbf')
    #     #     svc = svm.SVC(C=c, gamma=ga, kernel='poly')
    #     #     svc = svm.SVC(C=c * 0.5, gamma=ga, kernel='sigmoid')
    #         svc = svm.SVC(C=c * 0.5, gamma=ga, kernel='linear')
    #         svc.fit(X_train, Y_train)
    #         trainScore.append(svc.score(X_train, Y_train))
    #         testScore.append(svc.score(X_test, Y_test))
    #
    # plt.figure('svm')
    # plt.plot(trainScore, 'r.-')
    # plt.plot(testScore, 'b.-')
    # plt.legend(['train', 'test'])
    # plt.title('the score of svm')
    # plt.xlabel('c')
    # plt.ylabel('accuracy')
    # print("SVM testScore:" + str(max(testScore)))
    svclin = svm.SVC(C=0.5, gamma=2, kernel='linear')
    svclin.fit(X_train, Y_train)
    svcrbf = svm.SVC(C=0.5, gamma=2, kernel='rbf')
    svcrbf.fit(X_train, Y_train)
    return svclin.score(X_train, Y_train), svclin.score(X_test, Y_test), svcrbf.score(X_train, Y_train), svcrbf.score(X_test, Y_test)


def ANN(X_train, X_test, Y_train, Y_test):
    MLP = MLPClassifier(alpha=0.001, learning_rate_init=0.1,max_iter=100,hidden_layer_sizes=10).fit(X_train, Y_train)
    # print("ANN trainScore:" + str(MLP.score(X_train, Y_train)) + " testScore:" + str(MLP.score(X_test, Y_test)))
    return MLP.score(X_train, Y_train), MLP.score(X_test, Y_test)
    # model = Sequential()  # 建立模型
    # model.add(Dense(input_dim=5, output_dim=6))
    # model.add(Activation('relu'))  # 用relu函数作为激活函数，能够大幅提供准确度
    # model.add(Dense(input_dim=6, output_dim=6))
    # model.add(Activation('relu'))  # 用relu函数作为激活函数，能够大幅提供准确度
    # model.add(Dense(input_dim=6, output_dim=6))
    # model.add(Activation('relu'))  # 用relu函数作为激活函数，能够大幅提供准确度
    # model.add(Dense(input_dim=6, output_dim=1))
    # model.add(Activation('sigmoid'))  # 由于是0-1输出，用sigmoid函数作为激活函数
    #
    # model.compile(loss='binary_crossentropy', optimizer='adam', class_mode='binary')
    # # 编译模型。由于我们做的是二元分类，所以我们指定损失函数为binary_crossentropy，以及模式为binary
    # # 另外常见的损失函数还有mean_squared_error、categorical_crossentropy等，请阅读帮助文件。
    # # 求解方法我们指定用adam，还有sgd、rmsprop等可选
    #
    # model.fit(X_train, Y_train, nb_epoch=1000, batch_size=10)  # 训练模型，学习一千次
    # print("ANN trainScore:" + str(model.score(X_train, Y_train)) + " testScore:" + str(model.score(X_test, Y_test)))


def Xgboost(X_train, X_test, Y_train, Y_test):
    XGB = XGBClassifier().fit(X_train, Y_train)
    # print("XGboost trainScore:" + str(XGB.score(X_train, Y_train)) + " testScore:" + str(XGB.score(X_test, Y_test)))
    return XGB.score(X_train, Y_train), XGB.score(X_test, Y_test)


def Naive_bayes(X_train, X_test, Y_train, Y_test):
    NB_G = GaussianNB().fit(X_train, Y_train)
    # print("Gaussian Naive_bayes trainScore:" + str(NB_G.score(X_train, Y_train)) + " testScore:" + str(NB_G.score(X_test, Y_test)))
    NB_M = MultinomialNB().fit(X_train, Y_train)
    # print("Multinomial Naive_bayes trainScore:" + str(NB_M.score(X_train, Y_train)) + " testScore:" + str(NB_M.score(X_test, Y_test)))
    return NB_G.score(X_train, Y_train), NB_G.score(X_test, Y_test), NB_M.score(X_train, Y_train), NB_M.score(X_test, Y_test)



# def KNN(X_train, X_test, Y_train, Y_test):
#     n_neighbors = 20
#     K_N = neighbors.KNeighborsClassifier(n_neighbors, weights="distance")
#     K_N.fit(X_train, Y_train)
#     print("KNN testScore:" + str(K_N.score(X_test, Y_test)))

def RandomForest(X_train, X_test, Y_train, Y_test):
    RF = RandomForestClassifier(n_estimators=50, min_samples_leaf=2).fit(X_train, Y_train)
    # print("Random Forest trainScore:" + str(RF.score(X_train, Y_train)) + " testScore:" + str(RF.score(X_test, Y_test)))
    return RF.score(X_train, Y_train), RF.score(X_test, Y_test)


if __name__ == '__main__':
    data = pd.read_csv("../data/diabetes.csv")
    p_data = data.loc[data['Outcome'] == 1, :].sample(250)
    n_data = data.loc[data['Outcome'] == 0, :].sample(250)
    data = pd.concat([p_data,n_data])
    X = data.drop(columns='Outcome')
    Y = data["Outcome"]
    LR_TRAIN = []; LR_TEST = []
    SVMLIN_TRAIN = []; SVMLIN_TEST = []
    SVMRBF_TRAIN = []; SVMRBF_TEST = []
    ANN_TRAIN = []; ANN_TEST = []
    NBG_TRAIN = []; NBG_TEST = []
    NBM_TRAIN = []; NBM_TEST = []
    RF_TRAIN = []; RF_TEST = []
    for time in range(501):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
        lr_a_1, lr_a_2 = logisticRegression(X_train, X_test, Y_train, Y_test)
        LR_TRAIN.append(lr_a_1); LR_TEST.append(lr_a_2)
        # Decisiontree(X_train, X_test, Y_train, Y_test)
        svmlin_a_1, svmlin_a_2, svmrbf_a_1, svmrbf_a_2 = Svm(X_train, X_test, Y_train, Y_test)
        SVMLIN_TRAIN.append(svmlin_a_1); SVMLIN_TEST.append(svmlin_a_2)
        SVMRBF_TRAIN.append(svmrbf_a_1); SVMRBF_TEST.append(svmrbf_a_2)

        ann_a_1, ann_a_2 = ANN(X_train, X_test, Y_train, Y_test)
        ANN_TRAIN.append(ann_a_1); ANN_TEST.append(ann_a_2)
        # Xgboost(X_train, X_test, Y_train, Y_test)
        nbg_a_1, nbg_a_2, nbm_a_1, nbm_a_2 = Naive_bayes(X_train, X_test, Y_train, Y_test)
        NBG_TRAIN.append(nbg_a_1); NBG_TEST.append(nbg_a_2)
        NBM_TRAIN.append(nbm_a_1); NBM_TEST.append(nbm_a_2)
        # KNN(X_train, X_test, Y_train, Y_test)
        rf_a_1, rf_a_2 = RandomForest(X_train, X_test, Y_train, Y_test)
        RF_TRAIN.append(rf_a_1); RF_TEST.append(rf_a_2)

    print("LogisticRegression trainScore:" + str(np.average(np.array(LR_TRAIN))) + " testScore:" + str(np.average(np.array(LR_TEST))))
    print("SVMLIN trainScore:" + str(np.average(np.array(SVMLIN_TRAIN))) + " testScore:" + str(np.average(np.array(SVMLIN_TEST))))
    print("SVMRBF trainScore:" + str(np.average(np.array(SVMRBF_TRAIN))) + " testScore:" + str(np.average(np.array(SVMRBF_TEST))))
    print("ANN trainScore:" + str(np.average(np.array(ANN_TRAIN))) + " testScore:" + str(np.average(np.array(ANN_TEST))))
    print("RF trainScore:" + str(np.average(np.array(RF_TRAIN))) + " testScore:" + str(np.average(np.array(RF_TEST))))
    print("NBG trainScore:" + str(np.average(np.array(NBG_TRAIN))) + " testScore:" + str(np.average(np.array(NBG_TEST))))
    print("NBM trainScore:" + str(np.average(np.array(NBM_TRAIN))) + " testScore:" + str(np.average(np.array(NBM_TEST))))


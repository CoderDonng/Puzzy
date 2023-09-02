import math
import time

import matplotlib.pyplot as plt
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from common.Kmeans_plus import Centroids
from common.Input import Input
from common.Output import Output
from common.plot import PlotFun
from intervaltype2.IT2MF import IT2MF_Triangular
from intervaltype2.system.IT2_Antecedent import IT2_Antecedent
from intervaltype2.system.IT2_Consequent import IT2_Consequent
from intervaltype2.system.IT2_Rule import IT2_Rule
from intervaltype2.system.IT2_Rulebase import IT2_Rulebase
from type1.T1MF import T1MF_Triangular


# 参数优化部分
def getTurbIntervals(data, granularity):
    [centroids, stds] = Centroids(data, granularity).getCentroids()
    intervals_l = np.array(np.array(centroids) - 0.5 * np.array(stds)).tolist()
    intervals_r = np.array(np.array(centroids) + 0.5 * np.array(stds)).tolist()
    intervals = []
    for i in range(len(intervals_l)):
        intervals.append([intervals_l[i], intervals_r[i]])

    return intervals


def initialParamChromosome(supports, Graininesses):  # paramsChromosome: list
    fsParamsNum = 5
    paramChromosome = []
    for i in range(len(Graininesses)):
        fregment = [supports[i][0]] * 3
        arr = np.array(np.random.random(fsParamsNum * Graininesses[i] - 6) * (supports[i][1] - supports[i][0]) + supports[i][0]).tolist()
        arr.sort()
        temp_1 = arr[0:1]
        arr[0:1] = arr[2:3]
        arr[2:3] = temp_1
        temp_2 = arr[-2:-1]
        arr[-2:-1] = arr[-3:-4]
        arr[-3:-4] = temp_2
        fregment.extend(arr)
        fregment.extend([supports[i][1]] * 3)
        fregment.sort()
        paramChromosome.extend(fregment)

    return paramChromosome


def initialParamPop(supports, Graininesses, popSize, params):  # popMat: list
    curPopNum = 0
    popMat = []
    popMat.append(params)
    print("     Initial ParamPop......")
    while curPopNum < popSize - 1:
        chromosome = initialParamChromosome(supports, Graininesses)
        popMat.append(chromosome)
        curPopNum += 1

    return np.array(popMat)


def calParamCapacity(popMat, popSize, rulebase, Graininesses, X_train, Y_train, Antecedents, Consequents, attr_Input):
    print("     Calculate the fitness of param Chromosomes......")
    preP = []
    for curIndex in range(popSize):
        preP.append(getCCR_2(popMat[curIndex, :], rulebase, Graininesses, X_train, Y_train, Antecedents, Consequents,
                             attr_Input))

    bestACC = max(preP)
    bestParam = np.array(popMat[preP.index(bestACC), :])
    preP = np.array(preP).reshape(1, len(preP)) - min(preP)
    if sum(sum(preP)) == 0:
        p = np.ones(preP.shape[1]) * (1.0 / preP.shape[1])
    else:
        p = preP / sum(sum(preP))
    p = np.cumsum(p).tolist()
    return bestACC, bestParam, p


def getCCR_2(params, rulebase, Graininesses, X_train, Y_train, Antecedents, Consequents, attr_Input):
    # print("     parameters: " + str(params))

    for i in range(len(Graininesses) - 1):
        Antecedents[i][0].renew(
            [params[i * 15 + 0], params[i * 15 + 1], params[i * 15 + 2], params[i * 15 + 3], params[i * 15 + 4]])
        Antecedents[i][1].renew(
            [params[i * 15 + 5], params[i * 15 + 6], params[i * 15 + 7], params[i * 15 + 8], params[i * 15 + 9]])
        Antecedents[i][2].renew(
            [params[i * 15 + 10], params[i * 15 + 11], params[i * 15 + 12], params[i * 15 + 13], params[i * 15 + 14]])

    Consequents[0].renew([params[-10], params[-9], params[-8], params[-7], params[-6]])
    Consequents[1].renew([params[-5], params[-4], params[-3], params[-2], params[-1]])

    y_pred = []

    for num in range(X_train.shape[0]):
        for i in range(X_train.shape[1]):
            attr_Input[i].setInput(X_train[num, i])

        prob = rulebase.evaluate(1).get(output)
        fun_0 = Consequents[0].getMembershipFunction().getFSAverage(prob)
        fun_1 = Consequents[1].getMembershipFunction().getFSAverage(prob)
        if fun_0 < fun_1:
            y_pred.append(1)
        else:
            y_pred.append(0)

    score = 0
    for e in range(len(y_pred)):
        if y_pred[e] == Y_train[e]:
            score += 1

    score = score / len(y_pred)
    # print("     acc: " + str(score))
    return score


def selectFun_2(popMat, p, popSize):
    print("     Chromosomes selection......")
    tempMat = popMat.copy()
    P = [0]
    P.extend(p)
    for n in range(popSize):
        r = random.random()
        for m in range(len(P) - 1):
            if P[m] < r < P[m + 1]:
                popMat[n, :] = tempMat[m, :]
                break

    return popMat


def isTrueParam(paramChromosome, supports, Graininesses):
    Num = [0]
    Num.extend(np.array(np.cumsum(np.array(Graininesses) * 5)).tolist())
    for i in range(len(Graininesses)):
        Params = paramChromosome[Num[i]:Num[i + 1]]
        if Graininesses[i] == 3:
            if supports[i][0] > min(Params) or max(Params) > supports[i][1]:
                return False
            if not (Params[1] < Params[5] and Params[6] < Params[10]):
                return False
            if not (Params[2] < Params[7] < Params[12]):
                return False
            if not (Params[4] < Params[8] and Params[9] < Params[13]):
                return False
            if not (Params[0] <= Params[1] <= Params[2] <= Params[3] <= Params[4] and Params[1] < Params[3]):
                return False
            if not (Params[5] <= Params[6] <= Params[7] <= Params[8] <= Params[9] and Params[6] < Params[8]):
                return False
            if not (Params[10] <= Params[11] <= Params[12] <= Params[13] <= Params[14] and Params[11] < Params[13]):
                return False
        if Graininesses[i] == 2:
            if 0 > min(Params) or max(Params) > 1:
                return False
            if not (Params[1] < Params[5] and Params[2] < Params[7] and Params[4] < Params[8]):
                return False
            if not (Params[0] <= Params[1] <= Params[2] <= Params[3] <= Params[4] and Params[1] < Params[3]):
                return False
            if not (Params[5] <= Params[6] <= Params[7] <= Params[8] <= Params[9] and Params[6] < Params[8]):
                return False

    return True


def crossFun_2(popMat, popSize, Graininesses, beta, pc):
    print("     Chromosomes cross......")
    tempMat = popMat.copy()
    for i in range(0, popSize, 2):
        p = random.random()
        if p < pc:
            sec = np.random.randint(0, popMat.shape[1] - 1, 2)
            start = min(sec)
            end = max(sec)
            fatherGene = np.array(popMat[i, start:end])
            motherGene = np.array(popMat[i + 1, start:end])
            r = random.random()
            if r <= 0.5:
                c = math.pow(2 * r, 1.0 / (beta + 1))
            else:
                c = math.pow(2 * (1 - r), -1.0 / (beta + 1))
            popMat[i, start:end] = np.array(((1 + c) * motherGene + (1 - c) * fatherGene) / 2.0).tolist()
            popMat[i + 1, start:end] = np.array(((1 + c) * fatherGene + (1 - c) * motherGene) / 2.0).tolist()

            if not isTrueParam(popMat[i, :], supports, Graininesses):
                popMat[i, :] = tempMat[i, :]
                # print("          without cross")
            if not isTrueParam(popMat[i + 1, :], supports, Graininesses):
                popMat[i + 1, :] = tempMat[i + 1, :]
                # print("          without cross")

    return popMat


def heteroFun_2(popMat, popSize, supports, pm, n):
    print("     Chromosomes mutation......")
    tempMat = popMat.copy()
    for i in range(popSize):
        if random.random() < pm:
            point = random.randint(0, popMat.shape[1] - 1)
            heteroInterval = getParamsHeteroInterval(point, popMat[i, :], supports, Graininesses)
            r = random.random()
            if r <= 0.5:
                popMat[i, point] = popMat[i, point] + (math.pow(2 * r, 1 / (1 + n)) - 1) * (
                        popMat[i, point] - heteroInterval[0])
            else:
                popMat[i, point] = popMat[i, point] + (1 - math.pow(2 * (1 - r), 1 / (1 + n))) * (
                        heteroInterval[1] - popMat[i, point])

            if not isTrueParam(popMat[i, :], supports, Graininesses):
                popMat[i, :] = tempMat[i, :]
                # print("          without mutation")

    return popMat


def getParamsHeteroInterval(point, paramsChromosome, supports, Graininesses):
    Num = [0]
    Num.extend(np.array(np.cumsum(np.array(Graininesses) * 5)).tolist())

    for i in range(len(Graininesses)):

        if Num[i] <= point < Num[i + 1]:
            point = point - Num[i]
            Params = paramsChromosome[Num[i]: Num[i + 1]]
            if Graininesses[i] == 3:
                if point == 0:
                    # return [supports[i][0], Params[0]]
                    return [Params[0], Params[0]]
                elif point == 1:
                    # return [Params[0], min(Params[3], Params[4], Params[6])]
                    return [Params[1], Params[1]]
                elif point == 2:
                    # return [Params[0], min(Params[3], Params[6])]
                    return [Params[2], Params[2]]
                elif point == 3:
                    # return [max(Params[1],Params[2]), min(Params[7], Params[8])]
                    return [Params[2], Params[4]]
                elif point == 4:
                    return [Params[3], Params[8]]
                elif point == 5:
                    return [Params[1], Params[6]]
                elif point == 6:
                    return [Params[5], min(Params[7], Params[10])]
                elif point == 7:
                    return [max(Params[2], Params[6]), min(Params[12], Params[8])]
                elif point == 8:
                    return [max(Params[7], Params[4]), Params[9]]
                elif point == 9:
                    return [Params[8], Params[13]]
                elif point == 10:
                    return [Params[6], Params[11]]
                elif point == 11:
                    return [Params[10], Params[12]]
                elif point == 12:
                    return [Params[12], Params[12]]
                elif point == 13:
                    return [Params[13], Params[13]]
                elif point == 14:
                    # return [max(Params[9], Params[10]), min(Params[15], Params[16])]
                    return [Params[14], Params[14]]
            if Graininesses[i] == 2:
                if point == 0:
                    # return [supports[i][0], Params[0]]
                    return [Params[0], Params[0]]
                elif point == 1:
                    # return [Params[0], min(Params[3], Params[4], Params[6])]
                    return [Params[1], Params[1]]
                elif point == 2:
                    # return [Params[1], min(Params[3], Params[6])]
                    return [Params[2], Params[2]]
                elif point == 3:
                    # return [max(Params[1], Params[2]), min(Params[7], Params[8])]
                    return [Params[2], Params[4]]
                elif point == 4:
                    return [Params[3], Params[8]]
                elif point == 5:
                    return [Params[1], Params[6]]
                elif point == 6:
                    return [Params[5], Params[7]]
                elif point == 7:
                    return [Params[7], Params[7]]
                elif point == 8:
                    # return [max(Params[3], Params[4]), min(Params[9], Params[10])]
                    return [Params[8], Params[8]]
                elif point == 9:
                    # return [max(Params[5], Params[8]), Params[11]]
                    return [Params[9], Params[9]]


def optparams(Antecedents, Consequents, rulebase, base, params, Graininesses, X_train, Y_train, attr_Input):
    rulebase.clear()
    base = np.array(base).reshape(k, len(Graininesses))
    for i in range(base.shape[0]):
        antecedents = []
        if sum(base[i, :]) != 0:  # 排除为空的规则
            for j in range(len(base[i, :-1])):
                if base[i, j] != 0:
                    antecedents.append(Antecedents[j][base[i, j] - 1])
            consequent = Consequents[base[i, -1] - 1]
            rulebase.addRule(IT2_Rule(antecedents, consequent))

    popSize = 200  # 最大种群规模
    popMat = initialParamPop(supports, Graininesses, popSize, params)
    iterMax = 50  # 最大迭代次数
    pc = 0.9  # 交叉概率
    pm = 0.7  # 变异概率
    bestACC = 0  # 最高准确度
    bestParam = None
    # ACC_history = []
    iter = 0  # 当前迭代次数
    while iter < iterMax:
        print("     ---------- Iter " + str(iter + 1) + " -----------")
        iterBestCCR, iterBestParam, p = calParamCapacity(popMat, popSize, rulebase, Graininesses, X_train,
                                                         Y_train, Antecedents, Consequents, attr_Input)  # 计算适应度
        if bestACC < iterBestCCR:
            bestACC = iterBestCCR
            bestParam = iterBestParam

        # ACC_history.append(bestACC)
        print("     the best ACC is " + str(bestACC))

        mat = popMat.copy()
        # 选择
        popMat_afterSelection = selectFun_2(mat, p, popSize)
        # SBX交叉
        popMat_afterCross = crossFun_2(popMat_afterSelection, popSize, Graininesses, 2, pc)
        # polynomial mutation
        popMat_afterMutation = heteroFun_2(popMat_afterCross, popSize, supports, pm, 5)
        # 把每一代中的最优个体保留至下一代
        index = random.randint(0, popSize - 1)
        popMat_afterMutation[index, :] = bestParam
        # 经过选择、交叉和变异后生成新种群
        popMat = popMat_afterMutation

        iter += 1
    # plt.figure(1)
    # plt.plot(ACC_history, 'r.-')
    print(" the best params is " + str(bestParam) + "     the best ACC is " + str(bestACC))
    return bestParam, bestACC


# 规则优化部分
def optrulebase(Antecedents, Consequents, rulebase, rules, params, Graininesses, k, X_train, Y_train, attr_Input):

    popSize = 40  # 最大种群规模
    iterMax = 20  # 最大迭代次数
    popMat = initialbasePop(k, Graininesses, popSize, rules)  # 初始化种群
    pc = 0.8  # 交叉概率
    pm = 0.5  # 变异概率
    bestACC = 0  # 最高准确度
    bestBase = None
    ACC_history = []
    iter = 0  # 当前迭代次数

    while iter < iterMax:

        print("     ---------- Iter " + str(iter + 1) + " -----------")
        iterBestACC, iterBestBase, p = calBaseCapacity(popMat, popSize, k, Graininesses, X_train, Y_train, Antecedents,
                                                       Consequents, rulebase, attr_Input)  # 计算适应度
        if bestACC < iterBestACC:
            bestACC = iterBestACC
            bestBase = iterBestBase

        # ACC_history.append(bestACC)
        print("     the best ACC is " + str(bestACC))
        # 选择
        mat = popMat.copy()
        popMat_afterSelection = selectFun_1(mat, p, popSize)
        # 交叉
        popMat_afterCross = crossFun_1(popMat_afterSelection, popSize, k, Graininesses, pc)
        # 变异
        popMat_afterMutation = heteroFun_1(popMat_afterCross, popSize, k, Graininesses, pm)

        # 把每一代中的最优个体保留至下一代
        index = random.randint(0, popSize - 1)
        popMat_afterMutation[index, :] = bestBase
        # 经过选择、交叉和变异后生成新种群
        popMat = popMat_afterMutation

        iter += 1

    # plt.figure(2)
    # plt.plot(ACC_history, 'r.-')
    print("     the best rulebase is " + str(bestBase) + "     the best ACC is " + str(bestACC))
    rulebase.clear()

    return bestBase, bestACC


def initialbasePop(k, Graininesses, popSize, rules):
    curPopNum = 0
    populationMat = [rules]  # 初始化规则库时把原有的最优规则放进去，保证下一代的优化结果不会比之前的差
    print("     Initial basePop......")
    while curPopNum < popSize - 1:
        chromosome = initialbaseChromosome(k, Graininesses)
        if isTrueBase(chromosome, k, Graininesses):
            populationMat.append(chromosome)
            curPopNum += 1

    return np.array(populationMat)


def initialpseuRule(Graininesses):
    pesuRule = []
    for i in range(len(Graininesses)):
        # 每个属性下的细粒度水平加1，用于表示规则没使用到该属性的情况
        pesuRule.append(random.randint(0, Graininesses[i]))
    return pesuRule


def initialbaseChromosome(k, Graininesses):
    chromosome = []
    for i in range(k):
        pesuRule = initialpseuRule(Graininesses)
        chromosome.append(pesuRule)
    chromosome = np.array(chromosome).reshape(1, k * len(Graininesses))[0]

    return chromosome


def isTrueBase(pesuChromosome, k, Graininesses):
    chromosome = np.array(pesuChromosome).reshape(k, len(Graininesses))
    for i in range(chromosome.shape[0]):
        if sum(chromosome[i, 0:-1]) != 0 and chromosome[i, -1] == 0:
            return False
        elif sum(chromosome[i, 0:-1]) == 0 and chromosome[i, -1] != 0:
            return False

    return True


def getCCR_1(baseChromosome, k, Graininesses, X_train, Y_train, Antecedents, Consequents, rulebase, attr_Input):
    rulebase.clear()
    chromosome = np.array(baseChromosome).reshape(k, len(Graininesses))
    for i in range(chromosome.shape[0]):
        antecedents = []
        if sum(chromosome[i, :]) != 0:  # 排除为空的规则
            for j in range(len(chromosome[i, :-1])):
                if chromosome[i, j] != 0:
                    antecedents.append(Antecedents[j][chromosome[i, j] - 1])
            consequent = Consequents[chromosome[i, -1] - 1]
            rulebase.addRule(IT2_Rule(antecedents, consequent))

    y_pred = []
    for num in range(X_train.shape[0]):
        for c in range(X_train.shape[1]):
            attr_Input[c].setInput(X_train[num, c])

        rt = rulebase.evaluate(1).get(output)
        fun_0 = Consequents[0].getMembershipFunction().getFSAverage(rt)
        fun_1 = Consequents[1].getMembershipFunction().getFSAverage(rt)
        if fun_0 < fun_1:
            y_pred.append(1)
        else:
            y_pred.append(0)

    score = 0
    for e in range(len(y_pred)):
        if y_pred[e] == Y_train[e]:
            score += 1

    score = score / len(y_pred)
    # print("     rulebase: " + str(baseChromosome) + "     acc: " + str(score))
    return score


def calBaseCapacity(popMat, popSize, k, Graininesses, X_train, Y_train, Antecedents, Consequents, rulebase, attr_Input):
    print("     Calculate the fitness of rulebase Chromosomes......")
    preP = []
    for curIndex in range(popSize):
        preP.append(getCCR_1(popMat[curIndex, :], k, Graininesses, X_train, Y_train, Antecedents, Consequents, rulebase,
                             attr_Input))

    bestACC = max(preP)
    bestBase = popMat[preP.index(bestACC), :]
    preP = np.array(preP).reshape(1, len(preP)) - min(preP)
    if sum(sum(preP)) == 0:
        p = np.ones(preP.shape[1]) * (1.0 / preP.shape[1])
    else:
        p = preP / sum(sum(preP))

    p = np.cumsum(p).tolist()

    return bestACC, bestBase, p


# 选择
def selectFun_1(popMat, p, popSize):
    print("     Chromosomes selection......")
    tempMat = popMat.copy()
    P = [0]
    P.extend(p)
    for n in range(popSize):
        r = random.random()
        for m in range(1, len(P)):
            if P[m - 1] < r <= P[m]:
                popMat[n, :] = tempMat[m - 1, :]
                break

    return popMat


# 交叉
def crossFun_1(popMat, popSize, k, Graininesses, pc):
    print("     Chromosomes cross......")
    tempMat = popMat.copy()
    for i in range(0, popSize, 2):
        r = random.random()
        if r < pc:
            sec = np.random.randint(0, popMat.shape[1] - 1, 2)
            start = min(sec)
            end = max(sec)
            fatherGene = popMat[i, start:end]
            motherGene = popMat[i + 1, start:end]
            popMat[i, start:end] = motherGene
            popMat[i + 1, start:end] = fatherGene

            if not isTrueBase(np.array(popMat[i, :]).reshape(1, k * len(Graininesses))[0], k, Graininesses):
                popMat[i, :] = tempMat[i, :]

            if not isTrueBase(np.array(popMat[i + 1, :]).reshape(1, k * len(Graininesses))[0], k, Graininesses):
                popMat[i, :] = tempMat[i, :]

    return popMat


def heteroFun_1(popMat, popSize, k, Graininesses, pm):
    print("     Chromosomes mutation......")
    tempMat = popMat.copy()
    for i in range(popSize):
        if random.random() < pm:
            point = random.randint(0, popMat.shape[1] - 1)
            randRange = [num for num in range(Graininesses[point % len(Graininesses)] + 1)]
            randRange.remove(popMat[i, point])
            popMat[i, point] = random.choice(randRange)

            if not isTrueBase(np.array(popMat[i, :]).reshape(1, k * len(Graininesses))[0], k, Graininesses):
                popMat[i, :] = tempMat[i, :]

    return popMat


if __name__ == '__main__':
    data = pd.read_csv("../../data/diabetes.csv")
    p_data = data.loc[data['Outcome'] == 1, :].sample(250)
    n_data = data.loc[data['Outcome'] == 0, :].sample(250)
    data = pd.concat([p_data,n_data])
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
    k = 20  # 最大规则数
    rules = np.array(k * len(Graininesses) * [1])  # 最初规则库
    # 由CPA方法得到T1三角模糊数各个端点的扰动区间

    params = []
    for i in range(X_train.shape[1]):
        paramspart = []
        Turbs = getTurbIntervals(X_train[:, i].reshape(X_train.shape[0], 1), Graininesses[i])
        paramspart.extend([supports[i][0], supports[i][0]])
        paramspart.append(supports[i][0])
        paramspart.extend(Turbs[1])
        paramspart.extend(Turbs[0])
        paramspart.append(sum(Turbs[1]) / 2.0)
        paramspart.extend(Turbs[2])
        paramspart.extend(Turbs[1])
        paramspart.append(supports[i][1])
        paramspart.extend([supports[i][1], supports[i][1]])

        params.extend(paramspart)
    params.extend([0.0, 0.0, 0.0, 0.9, 1.0, 0.0, 0.1, 1.0, 1.0, 1.0])  # 最初参数

    # 构造各个attr的GCIT2型模糊数
    Attr_MFs = []
    for i in range(len(Graininesses) - 1):
        Attr_MFs.append([])
        Attr_MFs[i].append(IT2MF_Triangular("attr_" + str(i + 1) + "_level_1_mf",
                                            T1MF_Triangular("Upper", params[i * 15 + 0], params[i * 15 + 2],
                                                            params[i * 15 + 4]),
                                            T1MF_Triangular("Lower", params[i * 15 + 1], params[i * 15 + 2],
                                                            params[i * 15 + 3])))
        Attr_MFs[i].append(IT2MF_Triangular("attr_" + str(i + 1) + "_level_2_mf",
                                            T1MF_Triangular("Upper", params[i * 15 + 5], params[i * 15 + 7],
                                                            params[i * 15 + 9]),
                                            T1MF_Triangular("Lower", params[i * 15 + 6], params[i * 15 + 7],
                                                            params[i * 15 + 8])))
        Attr_MFs[i].append(IT2MF_Triangular("attr_" + str(i + 1) + "_level_3_mf",
                                            T1MF_Triangular("Upper", params[i * 15 + 10], params[i * 15 + 12],
                                                            params[i * 15 + 14]),
                                            T1MF_Triangular("Lower", params[i * 15 + 11], params[i * 15 + 12],
                                                            params[i * 15 + 13])))

    # 构建后件模糊数
    opt_MFs = []
    opt_MFs.append(IT2MF_Triangular("output_level_1_mf", T1MF_Triangular("Upper", params[-10], params[-8], params[-6]),
                                    T1MF_Triangular("Lower", params[-9], params[-8], params[-7])))
    opt_MFs.append(IT2MF_Triangular("output_level_2_mf", T1MF_Triangular("Upper", params[-5], params[-3], params[-1]),
                                    T1MF_Triangular("Lower", params[-4], params[-3], params[-2])))

    # 构建前件
    Antecedents = []
    for i in range(X_train.shape[1]):
        Antecedents.append([])
        for j in range(Graininesses[i]):
            Antecedents[i].append(IT2_Antecedent(None, Attr_MFs[i][j], attr_Input[i]))

    # 构建后件
    Consequents = []
    Consequents.append(IT2_Consequent(None, opt_MFs[0], output, None))
    Consequents.append(IT2_Consequent(None, opt_MFs[1], output, None))

    rulebase = IT2_Rulebase()

    time_start = time.time()
    rules, ACC_1 = optrulebase(Antecedents, Consequents, rulebase, rules, params, Graininesses, k, X_train, Y_train, attr_Input)
    params, ACC_2 = optparams(Antecedents, Consequents, rulebase, rules, params, Graininesses, X_train, Y_train, attr_Input)
    time_end = time.time()
    print("the time of optimization is " + str(time_end - time_start))
    np.savetxt("bestBase.txt", np.array(rules))
    np.savetxt("MFparams.txt", np.array(params))

    for i in range(len(Graininesses) - 1):
        Antecedents[i][0].renew(
            [params[i * 15 + 0], params[i * 15 + 1], params[i * 15 + 2], params[i * 15 + 3], params[i * 15 + 4]])
        Antecedents[i][1].renew(
            [params[i * 15 + 5], params[i * 15 + 6], params[i * 15 + 7], params[i * 15 + 8], params[i * 15 + 9]])
        Antecedents[i][2].renew(
            [params[i * 15 + 10], params[i * 15 + 11], params[i * 15 + 12], params[i * 15 + 13], params[i * 15 + 14]])

    Consequents[0].renew([params[-10], params[-9], params[-8], params[-7], params[-6]])
    Consequents[1].renew([params[-5], params[-4], params[-3], params[-2], params[-1]])

    rulebase.clear()
    rules = np.array(rules).reshape(k, len(Graininesses))
    for i in range(rules.shape[0]):
        antecedents = []
        if sum(rules[i, :]) != 0:  # 排除为空的规则
            for j in range(len(rules[i, :-1])):
                if rules[i, j] != 0:
                    antecedents.append(Antecedents[j][rules[i, j] - 1])
            consequent = Consequents[rules[i, -1] - 1]
            rulebase.addRule(IT2_Rule(antecedents, consequent))

    y_pred = []
    for num in range(X_test.shape[0]):
        for c in range(X_test.shape[1]):
            attr_Input[c].setInput(X_test[num, c])

        rt = rulebase.evaluate(1).get(output)
        fun_0 = Consequents[0].getMembershipFunction().getFSAverage(rt)
        fun_1 = Consequents[1].getMembershipFunction().getFSAverage(rt)
        if fun_0 < fun_1:
            y_pred.append(1)
        else:
            y_pred.append(0)

    score = 0
    for e in range(len(y_pred)):
        if y_pred[e] == Y_test[e]:
            score += 1

    score = score / len(y_pred)
    print("the accuracy on valid data is: " + str(score))


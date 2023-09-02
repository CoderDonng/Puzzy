import numpy
from datashape import Null

from type1.system.T1_Rule import T1_Rule


class T1_Rulebase:
    __rules:list
    __fStrengths:list
    __denominator: float
    __numerator: float
    __inferenceMethod = 1
    __implicationMethod = 1
    __PRODUCT = 0
    __MINIMUM = 1

    def __init__(self):
        # 构建新的规则库时，创建空列表：rules，空字典（hashMap）：outputSetBuffers、outputBuffers
        self.__rules = []
        self.__outputSetBuffers = {}
        self.__outputBuffers = {}

    def addRule(self, r:T1_Rule):  #添加规则 T1_Rule : ([T1_Antecedent],T1_Consequent)
        self.__rules.append(r)
        # 遍历每一条规则的后件
        for item in iter(r.getConsequents()): #通常只有一个后件，因此只执行一次
            # 若 原outputSetBuffers中已经包含该输出实例，则不添加，反之则添加至 outputSetBuffers、outputBuffers
            # outputSetBuffers结构 <Output,list>;  outputBuffers <Output,float>
            if item.getOutput() not in (self.__outputSetBuffers).keys():
                self.__outputSetBuffers[item.getOutput()] = numpy.zeros(item.getOutput().getDiscretisationLevel()).tolist()
                self.__outputBuffers[item.getOutput()] = Null

    def getNumberOfOutputs(self):
        return len(self.__outputSetBuffers)

    def getNumberOfRules(self):
        return len(self.__rules)

    def getInferenceMethod(self):
        if self.__inferenceMethod == 0:
            return "product"
        else:
            return "minimum"

    def setInferenceMethod(self, inferenceMethod):
        if inferenceMethod == 0:
            self.__inferenceMethod = 0
        else:
            if inferenceMethod != 1:
                raise Exception("Only product (0) and minimum (1) inference is currentlyt supported.")

            self.__inferenceMethod = 1


    def getImplicationMethod(self):
        if self.__implicationMethod == 0:
            return "product"
        else:
            return "minmum"

    def setImplicationMethod(self, implicationMethod):
        if implicationMethod == 0:
            self.__implicationMethod = 0
        else:
            if implicationMethod != 1:
                raise Exception("Only product (0) and minimum (1) implication is currentlyt supported.")

            self.__implicationMethod = 1

    def getRule(self, ruleNumber): return self.__rules[ruleNumber]

    def getInputs(self): return self.__rules[0].getInputs()

    def getOutputSetBuffers(self): return self.__outputSetBuffers

    def evaluate(self, defuzzificationType: int):
        if defuzzificationType == 0:
            return self.heightDefuzzification()
        elif defuzzificationType == 1:
            return self.centroidDefuzzification()
        else:
            raise Exception("The T1 evaluate() method only supports height defuzzification (0) and centroid defuzzification (1).")

    def centroidDefuzzification(self):
        tempHash_1 = self.__outputSetBuffers
        tempHash_2 = self.__outputBuffers
        for item in iter(tempHash_1.keys()):
            if len(tempHash_1[item]) == item.getDiscretisationLevel():
                tempHash_1[item] = numpy.zeros(len(tempHash_1[item])).tolist()
            else:
                tempHash_1[item] = numpy.zeros(item.getDiscretisationLevel()).tolist()

        self.__fStrengths = []
        for i in range(len(self.__rules)):
            self.__fStrengths.append(self.__rules[i].getFStrength(self.__implicationMethod))

            for Item in iter(self.__rules[i].getConsequents()):
                o = Item.getOutput()
                for j in range(o.getDiscretisationLevel()):
                    if self.__inferenceMethod == self.__PRODUCT:
                        tempHash_1[o][j] = max(tempHash_1.get(o)[j], self.__fStrengths[i] * Item.getMF().getFS(o.getDiscretizations()[j]))
                    else:
                        tempHash_1[o][j] = max(tempHash_1.get(o)[j], min(self.__fStrengths[i], Item.getMF().getFS(o.getDiscretizations()[j])))

        self.__numerator = 0.0
        self.__denominator = 0.0
        for it in iter(tempHash_2.keys()):
            self.__numerator = 0.0
            self.__denominator = 0.0
            for m in range(it.getDiscretisationLevel()):
                self.__numerator += it.getDiscretizations()[m] * tempHash_1[it][m]
                self.__denominator += tempHash_1[it][m]

            tempHash_2[it] = self.__numerator/self.__denominator

        self.__outputSetBuffers = tempHash_1
        self.__outputBuffers = tempHash_2

        return self.__outputBuffers

    def heightDefuzzification(self):
        # tempHash_1 <Output,list>;  tempHash_2 <Output,float> 只有一个后件，因此len(tempHash_1) 和 len(tempHash_2) = 1
        tempHash_1 = self.__outputSetBuffers
        tempHash_2 = self.__outputBuffers
        for item in iter(tempHash_1.keys()):
            if len(tempHash_1[item]) == 2:  #若该otempHash_1[item] 的 长度 为 2
                tempHash_1[item] = numpy.zeros(len(tempHash_1[item])).tolist()
            else:
                tempHash_1[item] = numpy.zeros(2).tolist()
        #创建一个列表存放规则的触发强度
        self.__fStrengths = []
        for i in range(len(self.__rules)):
            self.__fStrengths.append(self.__rules[i].getFStrength(self.__implicationMethod))
            for Item in iter(self.__rules[i].getConsequents()):
                o = Item.getOutput()
                # tempHash[0] 所有的规则强度*该规则后件对应的输出元素种类的隶属度函数的peak值之和
                Item_MF_Peak = Item.getMF().getPeak()
                tempHash_1[o][0] = tempHash_1[o][0] + self.__fStrengths[i]*Item_MF_Peak
                # tempHash[0] 所有的规则强度之和
                tempHash_1[o][1] = tempHash_1[o][1] + self.__fStrengths[i]

        for it in iter(tempHash_2.keys()):
            tempHash_2[it] = tempHash_1[it][0]/tempHash_1[it][1]

        self.__outputSetBuffers = tempHash_1
        self.__outputBuffers = tempHash_2

        return self.__outputBuffers

    def getRules(self): return self.__rules

    def changeRule(self, ruleToBeChanged: int, newRule: T1_Rule):
        self.__rules[ruleToBeChanged] = newRule

    def removeRule(self, ruleNumber: int):
        self.__rules.remove(ruleNumber)
import math

import numpy
from datashape import Null

from intervaltype2.IT2MF import IT2MF_Cylinder
from intervaltype2.operation.IT2Engine_Centroid import IT2Engine_Centroid
from intervaltype2.operation.IT2MF_Intersection import IT2MF_Intersection
from intervaltype2.operation.IT2MF_Union import IT2MF_Union
from intervaltype2.system.IT2_COSInferenceData import IT2_COSInferenceData
from intervaltype2.system.IT2_Rule import IT2_Rule
from common.Output import Output


class IT2_Rulebase:
    __rules: [IT2_Rule]
    __outputs:[Output]
    __showContext = False
    temp: IT2_Rule
    nan = float(math.nan)
    __CENTEROFSETS = 0
    __CENTROID = 1
    # __inferenceMethod = 1
    __implicationMethod = 1
    __PRODUCT = 0
    __MINIMUM = 1

    def __init__(self):
        self.__rules = []
        self.__outputs = []

    def getInputs(self):
        if len(self.__rules) == 0:
            raise Exception("No rules exist in the rulebase!")
        else:
            return self.__rules[0].getInputs()

    def getFuzzyLogicType(self):
        # 0: type-1, 1: interval type-2, 2: zSlices based general type-2
        return 1

    def addRule(self, r: IT2_Rule):
        self.__rules.append(r)
        for it in iter(r.getConsequents()):
            o = it.getOutput()
            if o not in self.__outputs:
                self.__outputs.append(o)

    def addRules(self,rules:[IT2_Rule]):
        for r in iter(rules):
            self.__rules.append(r)
            for it in iter(r.getConsequents()):
                o = it.getOutput()
                if o not in self.__outputs:
                    self.__outputs.append(o)

    def getRules(self): return self.__rules

    def getNumberOfRules(self): return len(self.__rules)

    def evaluateGetCentroid(self, typeReductionType: int):
        returnValue:{Output,[object]} = {}
        typeReductionOutput: {Output,[float,float]} = {}

        if typeReductionType == self.__CENTEROFSETS:
            typeReductionOutput = self.doCOSTypeReduction()
        elif typeReductionType == self.__CENTROID:
            typeReductionOutput = self.doReductionCentroid()

        for it in iter(self.__outputs):
            if typeReductionOutput.get(it) == Null:
                returnValue[it] = [Null, 1.0]
            else:
                returnValue[it] = [typeReductionOutput.get(it), 1.0]

        return returnValue

    def evaluate(self, typeReductionType: int):
        returnValue: {Output, float} = {}
        typeReductionOutput: {Output, [float, float]} = {}

        if typeReductionType == self.__CENTEROFSETS: # __CENTEROFSETS = 0
            typeReductionOutput = self.doCOSTypeReduction()
        elif typeReductionType == self.__CENTROID: # __CENTROID = 1
            typeReductionOutput = self.doReductionCentroid()

        for it in iter(self.__outputs):
            if typeReductionOutput.get(it) == Null:
                returnValue[it] = 0.0
            else:
                returnValue[it] = (typeReductionOutput.get(it)[0]+typeReductionOutput.get(it)[1])/2

        return returnValue

    def doCOSTypeReduction(self):
        returnValue: {Output, [object]} = {}
        data = self.getFiringIntervalsForCOS_TR()
        data_1 = []
        data_2 = []
        for item in data.items():
            data_1.append(item[0])
            data_2.append(item[1])

        if len((data_2[0])[0]) == 0:
            for output in iter(data.keys()):
                returnValue[output] = Null
            return returnValue
        else:
            for i in range(len(data_1)):
                leftData = data_2[i][0]
                rightData = data_2[i][1]

                fir = numpy.zeros(len(leftData)).tolist()
                yr = 0.0; yl = 0.0; yDash = 0.0; yDashDash = 0.0

                R: int = 0; L: int = 0
                l_endpoints = []; r_endpoints = []
                for r in range(2):
                    stopFlag = False
                    for j in range(len(fir)):
                        fir[j] = (rightData[j].getFStrength()[0]+rightData[j].getFStrength()[1])/2
                    if r == 0:
                        for it in iter(leftData):
                            l_endpoints.append(it.getSelectedCentroidEndpoint())
                        yl = self.weightedSigma(fir, l_endpoints)
                        yDash = yl
                    else:
                        for it in iter(rightData):
                            r_endpoints.append(it.getSelectedCentroidEndpoint())
                        yr = self.weightedSigma(fir,r_endpoints)
                        yDash = yr

                    while not stopFlag:
                        if r == 0:
                            for j in range(len(fir) - 1):
                                if l_endpoints[j] <= yDash <= l_endpoints[j + 1]:
                                    L = j
                                    break

                            for j in range(L + 1):
                                fir[j] = leftData[j].getFStrength()[1]
                            for j in range(L + 1, len(fir)):
                                fir[j] = leftData[j].getFStrength()[0]

                            yl = self.weightedSigma(fir, l_endpoints)
                            if yl == self.nan:
                                yl = 0
                                break
                            yDashDash = yl

                            if abs(yDash) - abs(yDashDash) < 0.000001:
                                stopFlag = True
                                yDashDash = yl
                            else:
                                yDash = yDashDash

                        else:
                            for j in range(len(fir)-1):
                                if r_endpoints[j] <= yDash <= r_endpoints[j + 1]:
                                    R = j
                                    break

                            for j in range(R+1):
                                fir[j] = rightData[j].getFStrength()[0]
                            for j in range(R+1,len(fir)):
                                fir[j] = rightData[j].getFStrength()[1]

                            if len(fir) == 1 and fir[0] == 0: fir[0] = 0.00001
                            yr = self.weightedSigma(fir,r_endpoints)
                            yDashDash = yr

                            if abs(yDash - yDashDash) < 0.000001:
                                stopFlag = True
                                yDashDash = yr
                            else:
                                yDash = yDashDash

                returnValue[data_1[i]] = [yl, yr]
            return returnValue

    def getFiringIntervalsForCOS_TR(self):
        returnValue:{Output,[object]} = {}

        for i in range(len(self.__rules)): #遍历规则
            ruleCons = self.__rules[i].getConsequents() #获取规则后件
            firingStrength = self.__rules[i].getFStrength(self.__implicationMethod) # __implicationMethod = 1 计算规则触发强度 使用MINIMUM计算方式，得到点火区间
            print("fStrength of rule "+str(i)+" is "+str(firingStrength))
            if firingStrength[1] > 0.0: #firingStrength[1] 记录的是upper mf的值，若firingStrength[1]小于0，则该规则的必定无法触发
                for c in iter(ruleCons):
                    if c.getOutput() not in returnValue.keys():
                        returnValue[c.getOutput()] = [[],[]]
                    returnValue.get(c.getOutput())[0].append(IT2_COSInferenceData(firingStrength, self.__rules[i].getConsequentCentroid(c.getOutput())[0]))
                    returnValue.get(c.getOutput())[1].append(IT2_COSInferenceData(firingStrength, self.__rules[i].getConsequentCentroid(c.getOutput())[1]))

        return returnValue

    def doReductionCentroid(self):
        overallOutputSet = {}
        firstFiredForOutput:{Output,bool} = {}

        for output in iter(self.__outputs):
            firstFiredForOutput[output] = True

        for i in range(len(self.__rules)):
            fStrength = self.__rules[i].getFStrength(self.__implicationMethod) # implicationMethod = 1，使用MINIMUM t-Norm计算，implicationMethod = 0，使用PRODUCT t-Norm计算
            # print("fStrength of rule " + str(i) + " is " + str(fStrength))
            if fStrength[1] > 0.0:
                for con in iter(self.__rules[i].getConsequents()):
                    o = con.getOutput()
                    if firstFiredForOutput.get(o):
                        overallOutputSet[o] = IT2MF_Intersection(IT2MF_Cylinder("FiringInterval",fStrength,None,None), con.getMembershipFunction())
                        if not overallOutputSet.get(o).intersectionExists():
                            overallOutputSet[o] = Null
                        firstFiredForOutput[o] = False
                    else:
                        if overallOutputSet.get(o) == Null:
                            overallOutputSet[o] = IT2MF_Intersection(IT2MF_Cylinder("FiringInterval", fStrength, None, None), con.getMembershipFunction())
                            if not overallOutputSet.get(o).intersectionExists():
                                overallOutputSet[o] = Null
                        else:
                            overallOutputSet[o] = IT2MF_Union(IT2MF_Intersection(IT2MF_Cylinder("FiringInterval", fStrength, None, None), con.getMembershipFunction()),overallOutputSet.get(o))

        IT2EC = IT2Engine_Centroid(100)
        returnValue:{Output,[float,float]} = {}
        for op in iter(self.__outputs):
            IT2EC.setPrimaryDiscretizationLevel(op.getDiscretisationLevel())
            returnValue[op] = IT2EC.getCentroid(overallOutputSet.get(op))

        return returnValue

    def weightedSigma(self, w:list, y:list):
        numerator = 0.0
        denominator =0.0
        for i in range(len(w)):
            numerator += w[i]*y[i]
            denominator += w[i]

        if denominator == 0.0:
            return 0.0
        else:
            return numerator/denominator

    def removeRule(self, ruleNumber: int):
        self.__rules.remove(ruleNumber)

    def getImplicationMethod(self):
        if self.__implicationMethod == self.__PRODUCT:
            return "PRODUCT"
        else:
            return "MINIMUM"

    def setImplicationMethod(self, implicationMethod):
        if implicationMethod == self.__PRODUCT:
            self.__implicationMethod = self.__PRODUCT
        elif implicationMethod == self.__MINIMUM:
            self.__implicationMethod = self.__MINIMUM
        else:
            raise Exception("Only product (0) and minimum (1) implication is currentlyt supported.")

    def clear(self):
        self.__rules.clear()



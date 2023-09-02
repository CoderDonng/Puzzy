from datashape import Null

from common.Output import Output
from intervaltype2.system.IT2_Rulebase import IT2_Rulebase
from type2.T2MF import T2MF_Prototype
from type2.operation.T2MF_Intersection import T2MF_Intersection
from type2.system.T2Engine_Intersection import T2Engine_Intersection
from type2.system.T2Engine_Union import T2Engine_Union
from type2.system.T2_Antecedent import T2_Antecedent
from type2.system.T2_Rule import T2_Rule


class T2_Rulebase:

    __rules: [T2_Rule]
    __T2EU: T2Engine_Union
    __T2EI: T2Engine_Intersection
    __outputs: [Output]
    __output = Null
    __CENTEROFSETS = 0
    __CENTROID = 1
    __implicationMethod = 1
    __PRODUCT = 0
    __MINIMUM = 1
    __showContext = False

    def __init__(self, initialNumberOfRules: int):
        self.__rules = [Null for i in range(initialNumberOfRules)]
        self.__T2EU = T2Engine_Union()
        self.__T2EI = T2Engine_Intersection()
        self.__outputs = []
        self.__rules = []

    def addrule(self, rule: T2_Rule):
        self.__rules.append(rule)
        for it in iter(rule.getConsequents()):
            o = it.getOutput()
            if o not in self.__outputs:
                self.__outputs.append(o)

    def addrules(self, r: [T2_Rule]):
        for it_1 in iter(r):
            self.__rules.append(it_1)
            for it_2 in iter(it_1.getConsequents()):
                o = it_2.getOutput()
                if o not in self.__outputs:
                    self.__outputs.append(o)

    def getRules(self): return self.__rules

    def getFuzzyLogicType(self): return 2

    def get_T2Engine_Intersection(self): return self.__T2EI

    def get_T2Engine_Union(self): return self.__T2EU

    def getOverallOutput(self):
        returnValue: {Output, T2MF_Prototype} = {}
        for r in range(len(self.__rules)):
            temp = self.__rules[r].getRawOutput()
            for it in iter(self.__outputs):
                if r == 0:
                    returnValue[it] = temp.get(it)
                else:
                    returnValue[it] = self.__T2EU.getUnion(returnValue.get(it), temp.get(it))

        return returnValue

    def evaluateGetCentroid(self, typeReductionType: int):

        returnValue:{Output, []} = {}
        IT2_RBS = self.getIT2Rulebases()

        zValues = self.__rules[0].getAntecedents()[0].getSet().getZValues()

        temp: {Output, []}
        for i in range(len(IT2_RBS)):
            temp = IT2_RBS[i].evaluateGetCentroid(typeReductionType)
            for it in iter(temp.keys()):
                if i == 0:
                    returnValue[it] = [[Null for i in range(len(IT2_RBS))], [Null for i in range(len(IT2_RBS))]]
                (returnValue[it][0])[i] = temp.get(it)[0]
                (returnValue[it][1])[i] = zValues[i]

        return returnValue

    def evaluate(self, typeReductionType: int):
        returnValue: {Output, float} = {}
        IT2_RBS = self.getIT2Rulebases()
        rawOutputValues = [Null for i in range(len(IT2_RBS))]
        for i in range(len(IT2_RBS)):
            print("                                        ############  calculate the slice ["+str(i)+"] ############")
            rawOutputValues[i] = IT2_RBS[i].evaluate(typeReductionType)

        zValues = self.__rules[0].getAntecedents()[0].getSet().getZValues()

        for it in iter(self.__outputs):
            i = 0
            numerator = 0.0
            denominator = 0.0
            for it_2 in iter(rawOutputValues):
                numerator += it_2.get(it) * zValues[i]
                denominator += zValues[i]
                i += 1

            returnValue[it] = numerator / denominator

        return returnValue

    def getIT2Rulebases(self):
        RBS = [Null for i in range(self.__rules[0].getAntecedents()[0].getSet().getNumberOfSlices())]
        for i in range(len(RBS)):
            RBS[i] = IT2_Rulebase()
            for j in range(self.getNumberOfRules()):
                RBS[i].addRule(self.__rules[j].getRuleasIT2Rules()[i])

            RBS[i].setImplicationMethod(self.__implicationMethod)

        return RBS

    def getRule(self, number: int): return self.__rules[number]

    def changeRule(self, ruleToBeChanged: int, newRule: T2_Rule): self.__rules[ruleToBeChanged] = newRule

    def removeRule(self, ruleNumber): self.__rules.remove(ruleNumber)

    def getNumberOfRules(self): return len(self.__rules)

    def containsRule(self, rule: T2_Rule): return (rule in self.__rules)

    def getRulesWithAntecedents(self, antecedents: [T2_Antecedent]):
        matchs = []
        for i in range(len(self.__rules)):
            if self.__rules[i].getAntecedents() == antecedents:
                matchs.append(self.__rules[i])
        return matchs

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



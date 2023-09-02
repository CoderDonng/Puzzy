from datashape import Null

from common.Input import Input
from common.Output import Output
from intervaltype2.system.IT2_Rule import IT2_Rule
from type1.operation.T1MF_Meet import T1MF_Meet
from type2.T2MF import T2MF_CylExtension, T2MF_Prototype
from type2.operation.T2MF_Intersection import T2MF_Intersection
from type2.system.T2Engine_Intersection import T2Engine_Intersection
from type2.system.T2_Antecedent import T2_Antecedent


class T2_Rule:
    __antecedents:[]
    __consequents:{}
    __fs:T1MF_Meet
    __cylExt: T2MF_CylExtension
    __EI: T2Engine_Intersection

    def __init__(self, antecedents:[T2_Antecedent], consequent):
        self.__antecedents = antecedents
        self.__consequents = {}
        self.__EI = T2Engine_Intersection()
        if type(consequent) is list:
            for i in range(len(consequent)):
                self.__consequents[consequent[i].getOutput()] = consequent[i]
        else:
            self.__consequents[consequent.getOutput()] = consequent

    def getFS(self):
        if len(self.__antecedents) == 1:
            return self.__antecedents[0].getFS()
        else:
            self.__fs = T1MF_Meet(self.__antecedents[0].getFS(), self.__antecedents[1].getFS())
            if not self.__fs.intersectionExists():
                self.__fs = Null
            else:
                for i in range(2, len(self.__antecedents)):
                    self.__fs = T1MF_Meet(self.__fs, self.__antecedents[i].getFS())

            if self.__fs is not Null and self.__fs.intersectionExists():
                return self.__fs
            else:
                return Null

    def getInputs(self):
        inputs = []
        for i in range(self.getNumberOfAntecedents()):
            inputs.append(self.getAntecedents()[i].getInput())
        return inputs

    def getOutput(self):
        return self.getConsequents()[0].getOutput()

    def getRawOutput(self):
        returnValue: {Output,T2MF_Intersection} = {}
        baseSet = self.getFS()
        for it in iter(self.__consequents.values()):
            o = it.getOutput()
            if baseSet is not Null:
                self.__cylExt = T2MF_CylExtension(baseSet,self.__antecedents[0].getSet().getNumberOfSlices())
                returnValue[o] = self.__EI.getIntersection(self.__cylExt, it.getSet())
            else:
                returnValue[o] = self.__EI.getIntersection(self.__cylExt, Null)

        return returnValue

    def getRuleasIT2Rules(self):
        rs = []
        for i in range(self.getAntecedents()[0].getSet().getNumberOfSlices()):
            ans = []
            cons = []
            for a in range(len(self.__antecedents)):
                IT2s = self.getAntecedents()[a].getAntecedentasIT2Sets()
                ans.append(IT2s[i])
                if isinstance(self.getAntecedents()[a].getInput().getInputMF(), T2MF_Prototype):
                    temp = ans[a].getInput()
                    domain = temp.getDomain()
                    inputName = temp.getName()
                    mf = temp.getInputMF().getZSlice(i)
                    newInput = Input(inputName, domain)
                    newInput.setInputMF(mf)
                    ans[a].setInput(newInput)

            for it in iter(self.__consequents.values()):
                IT2Set = it.getConsequentsIT2Sets()
                cons.append(IT2Set[i])

            rs.append(IT2_Rule(ans, cons))

        return rs


    def getAntecedents(self):
        return self.__antecedents

    def getConsequents(self):
        cons =[]
        for i in iter(self.__consequents.values()):
            cons.append(i)
        return cons

    def getNumberOfAntecedents(self): return len(self.__antecedents)

    def getNumberOfConsequents(self): return len(self.__consequents)

    def equals(self, rule):
        if self == rule: return True
        if not isinstance(rule, T2_Rule):
            return False
        else:
            isEqual = True
            for an_1 in iter(self.getAntecedents()):
                temp = False
                for an_2 in iter(rule.getAntecedents()):
                    if an_1 == an_2:
                        temp = True

                isEqual &= temp
            for con_1 in iter(self.getConsequents()):
                temp =False
                for con_2 in iter(rule.getConsequents()):
                    if con_1 == con_2:
                        temp = True

                isEqual &= temp

        return isEqual

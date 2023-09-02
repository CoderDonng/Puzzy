from datashape import Null

from type1.T1MF import T1MF_Singleton


class T1_Rule():
    __antecedents :[[]]
    __consequents :dict
    __PRODUCT = 0
    __MINIMUM = 1

    def __init__(self, antecedents, consequents):
        self.__antecedents = antecedents
        self.__consequents = {}
        if type(consequents) == list:
            for i in range(len(consequents)):
                self.__consequents[consequents[i].getOutput()] = consequents[i]
        else:
            self.__consequents[consequents.getOutput()] = consequents

    def getNumberOfAntecedents(self):
        num = 0
        for it in iter(self.__antecedents):
            num += len(it)
        return num

    def getNumberOfConsequents(self):
        return len(self.__consequents)

    def getAntecedents(self):
        return self.__antecedents

    def getConsequents(self):
        cons = []
        for i in self.__consequents.values():
            cons.append(i)
        return cons

    def getInputs(self):
        inputs = []
        for i in range(self.getNumberOfAntecedents()):
            inputs.append(self.__antecedents[i].getInput)

        return inputs

    def compareBasedOnAntecedents(self, r):
        if len(self.getAntecedents()) == len(r.getAntecedents()): #若取并集的前件个数相等
            for i in range(len(self.getAntecedents())):
                if len(self.getAntecedents()[i]) == len(r.getAntecedents()[i]): #检查每个对应的并集中的前件个数是否相等
                    for j in range(len(self.getAntecedents()[i])):
                        if self.getAntecedents()[i][j] != r.getAntecedents()[i][j]:
                            return False

            return True
        else:
            return False

    def getFStrength(self, tNorm): #获取规则触发强度
        xmax:float
        fs = []
        if tNorm == 0: #取小时使用PRODUCT算子
            for i in range(len(self.getAntecedents())):
                fStrength = 1.0
                for j in range(len(self.getAntecedents()[i])):
                    # 该test中的Input的inputMF都为Singleton
                    if isinstance(self.__antecedents[i][j].getInput().getInputMF(), T1MF_Singleton):
                        fStrength *= self.__antecedents[i][j].getFS(Null)
                        # 以badFood前件为例，执行T1MF_Triangular类中的getFS(0) =1
                    else:
                        xmax = self.__antecedents[i][j].getMax(0)
                        fStrength *= self.__antecedents[i][j].getInput().getInputMF().getFS(xmax)*self.__antecedents[i][j].getFS(xmax)

                fs.append(fStrength)
        else:
            for i in range(len(self.getAntecedents())):
                fStrength = 1.0
                for j in range(len(self.getAntecedents()[i])):
                    if isinstance(self.__antecedents[i][j].getInput().getInputMF(), T1MF_Singleton):
                        fStrength = min( fStrength, self.__antecedents[i][j].getFS(Null))
                    else:
                        xmax = self.__antecedents[i][j].getMax(1)
                        fStrength *= min(fStrength, min(self.__antecedents[i][j].getInput().getInputMF().getFS(xmax), self.__antecedents[i][j].getFS(xmax)))

                fs.append(fStrength)
        return max(fs)


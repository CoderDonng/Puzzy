from datashape import Null

from intervaltype2.system.IT2_Antecedent import IT2_Antecedent
from intervaltype2.system.IT2_Consequent import IT2_Consequent
from common.Output import Output
from type1.T1MF import T1MF_Singleton, T1MF_Prototype


class IT2_Rule:
    __antecedents: [IT2_Antecedent]
    __consequents: {Output,IT2_Consequent}
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

    def getFStrength(self, tNorm): #获取规则触发强度
        fStrength = [1.0, 1.0] #分别记录lower隶属度函数和upper隶属度函数的触发值
        if tNorm == self.__PRODUCT: # __PRODUCT = 0

            for i in range(len(self.getAntecedents())):
                for j in range(self.getAntecedents()[i]):
                    # 该test中的Input的inputMF都为Singleton
                    if isinstance(self.__antecedents[i][j].getInput().getInputMF(), T1MF_Singleton):
                        fStrength[0] = fStrength[0] * self.__antecedents[i][j].getFS(Null)[0]
                        fStrength[1] = fStrength[1] * self.__antecedents[i][j].getFS(Null)[1]
                        # 以badFood前件为例，执行T1MF_Triangular类中的getFS(0) =1
                    elif isinstance(self.__antecedents[i].getInput().getInputMF(), T1MF_Prototype):
                        xmax = self.__antecedents[i].getMax(self.__PRODUCT)
                        fStrength[0] = fStrength[0] * self.__antecedents[i].getMF().getLMF().getFS(xmax[0])*self.__antecedents[i][j].getInput().getInputMF().getFS(xmax[0])
                        fStrength[1] = fStrength[1] * self.__antecedents[i].getMF().getUMF().getFS(xmax[1]) * \
                                       self.__antecedents[i][j].getInput().getInputMF().getFS(xmax[1])
                    else: #当InputMF为IT2型
                        xmax = self.__antecedents[i].getMax(self.__PRODUCT)
                        fStrength[0] = fStrength[0] * self.__antecedents[i][j].getMF().getLMF().getFS(xmax[0]) * \
                                       self.__antecedents[i][j].getInput().getInputMF().getLMF().getFS(xmax[0])
                        fStrength[1] = fStrength[1] * self.__antecedents[i][j].getMF().getUMF().getFS(xmax[1]) * \
                                       self.__antecedents[i][j].getInput().getInputMF().getUMF().getFS(xmax[1])
        else: # __MINIMUM = 1
            for i in range(len(self.__antecedents)):
                if isinstance(self.__antecedents[i].getInput().getInputMF(), T1MF_Singleton):
                    fStrength[0] = min( fStrength[0], self.__antecedents[i].getFS(Null)[0])
                    fStrength[1] = min(fStrength[1], self.__antecedents[i].getFS(Null)[1])
                elif isinstance(self.__antecedents[i].getInput().getInputMF(), T1MF_Prototype):
                    xmax = self.__antecedents[i].getMax(self.__MINIMUM)
                    fStrength[0] = min(fStrength[0], min(self.__antecedents[i].getMF().getLMF().getFS(xmax[0]), self.__antecedents[i].getInput().getInputMF().getFS(xmax[0])))
                    fStrength[1] = min(fStrength[1], min(self.__antecedents[i].getMF().getUMF().getFS(xmax[1]),
                                                         self.__antecedents[i].getInput().getInputMF().getFS(xmax[1])))
                else:
                    xmax = self.__antecedents[i].getMax(self.__MINIMUM)
                    fStrength[0] = min(fStrength[0], min(self.__antecedents[i].getMF().getLMF().getFS(xmax[0]),
                                                         self.__antecedents[i].getInput().getInputMF().getLMF().getFS(xmax[0])))
                    fStrength[1] = min(fStrength[1], min(self.__antecedents[i].getMF().getUMF().getFS(xmax[1]),
                                                         self.__antecedents[i].getInput().getInputMF().getUMF().getFS(xmax[1])))
        return fStrength

    def getInputs(self):
        inputs = []
        for i in range(self.getNumberOfAntecedents()):
            inputs.append(self.__antecedents[i].getInput)

        return inputs

    def getConsequentCentroid(self, o:Output): return self.__consequents.get(o).getCentroid()
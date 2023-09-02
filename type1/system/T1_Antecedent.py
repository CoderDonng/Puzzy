from datashape import Null

from common.Input import Input
from type1.T1MF import T1MF_Gaussian


class T1_Antecedent:
    __name:str
    __input: Input

    def __init__(self, name: str, mF, input: Input):
        self.__name = name
        self.__input = input
        self.__mF = mF

    def setMF(self, mF): self.__mF = mF

    def getMF(self): return self.__mF

    def getInput(self): return self.__input

    def getName(self): return self.__name

    def setName(self, name:str): self.__name = name

    def getFS(self, x):
        if x is Null:
            return self.__mF.getFS(self.__input.getInput())
        else:
            return self.__mF.getFS(x)

    def getMax(self, tNorm:int):
        xmax =0.0
        if isinstance(self.__input.getInputMF(),T1MF_Gaussian) and isinstance(self.getMF(),T1MF_Gaussian):
            gaussian = self.__input.getInputMF()
            sigmaX = gaussian.getSpread()
            meanX = gaussian.getMean()
            antecedentMF = self.__mF

            sigmaF = antecedentMF.getSpread()
            meanF = antecedentMF.getMean()

            if tNorm == 0:
                xmax = (sigmaX * sigmaX * meanF + sigmaF * sigmaF * meanX) / (sigmaX * sigmaX + sigmaF * sigmaF)
            else:
                xmax = (sigmaX * meanF + sigmaF * meanX) / (sigmaX + sigmaF)

        else:
            valxmax = 0.0
            domain = self.__input.getDomain()[1] - self.__input.getDomain()[0]
            incr = domain / (domain * 50) # 0.02
            x = 0.0
            temp = 0.0
            i = 0
            while i <= domain*50:
                if tNorm == 0:
                    temp = self.__input.getInputMF().getFS(x)*self.getMF().getFS(x)
                else:
                    temp = min(self.__input.getInputMF().getFS(x), self.getMF().getFS(x))
                if temp >= valxmax:
                    valxmax = temp
                    xmax = x
                x = x + incr
                i += 1

        return xmax

    def reverse(self):
        return T1_Antecedent_reverse(self.__name, self.__mF, self.__input)


class T1_Antecedent_reverse(T1_Antecedent):
    def __init__(self, name: str, mF, input: Input):
        super().__init__(name, mF, input)

    def getFS(self, x):
        return 1- super().getFS(x)



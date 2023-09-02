from datashape import Null

from intervaltype2.IT2MF import IT2MF_Prototype, IT2MF_Triangular, IT2MF_Gauangle
from common.Input import Input
from type1.T1MF import T1MF_Prototype


class IT2_Antecedent:
    __name:str
    __input: Input

    def __init__(self, name: str, mF, input: Input):
        self.__input = input
        self.__mF = mF
        if name is None:
            self.__name = self.__mF.getName()
        else:
            self.__name = name

    def getMF(self): return self.__mF

    def getInput(self): return self.__input

    def getName(self): return self.__name

    def setName(self, name:str): self.__name = name

    def getFS(self, x):
        if x is Null:
            return self.__mF.getFS(self.__input.getInput())
        else:
            return self.__mF.getFS(x)

    def setInput(self,input):
        self.__input = input

    def getSet(self): return self.__mF



    def getMax(self, tNorm:int):
        xmax = [0.0, 0.0]
        l_xmax = 0; u_xmax = 0
        domain = self.__input.getDomain()[1] - self.__input.getDomain()[0]
        incr = 1.0/50.0
        x = 0.0
        l_temp = 0
        u_temp = 0
        if isinstance(self.__input.getInputMF(),T1MF_Prototype):
            for i in range(int(domain*50)+1):
                if tNorm == 0:
                    l_temp = self.__input.getInputMF().getFS(x) * self.getMF().getLMF().getFS(x)
                    u_temp = self.__input.getInputMF().getFS(x) * self.getMF().getUMF().getFS(x)
                else:
                    l_temp = min(self.__input.getInputMF().getFS(x), self.getMF().getLMF().getFS(x))
                    u_temp = min(self.__input.getInputMF().getFS(x), self.getMF().getUMF().getFS(x))

                if l_temp >= l_xmax:
                    l_xmax = l_temp
                    xmax[0] = x

                if u_temp >= u_xmax:
                    u_xmax = u_temp
                    xmax[1] = x

                x += incr

        elif isinstance(self.__input.getInputMF(),IT2MF_Prototype):
            for i in range(int(domain * 50) + 1):
                if tNorm == 0:
                    l_temp = self.__input.getInputMF().getFS(x)[0] * self.getMF().getLMF().getFS(x)
                    u_temp = self.__input.getInputMF().getFS(x)[1] * self.getMF().getUMF().getFS(x)
                else:
                    l_temp = min(self.__input.getInputMF().getFS(x)[0], self.getMF().getLMF().getFS(x))
                    u_temp = min(self.__input.getInputMF().getFS(x)[1], self.getMF().getUMF().getFS(x))

                if l_temp >= l_xmax:
                    l_xmax = l_temp
                    xmax[0] = x

                if u_temp >= u_xmax:
                    u_xmax = u_temp
                    xmax[1] = x

                x += incr

        return xmax

    def renew(self, params):
        self.__mF.renewparams(params)


class IT2_Antecedent_reverse(IT2_Antecedent):

    def __init__(self, name: str, mF, input: Input):
        super().__init__(name, mF, input)

    def getFS(self, x):
        super().getFS(x)


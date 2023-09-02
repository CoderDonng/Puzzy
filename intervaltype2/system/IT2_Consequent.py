from intervaltype2.operation.IT2Engine_Centroid import IT2Engine_Centroid
from common.Output import Output


class IT2_Consequent:
    __name: str
    __mF = None
    __centroid:[float,float]
    __output: Output

    def __init__(self, name, m, output: Output, centroid):
        if centroid is None:
            self.__mF = m
            if name is not None:
                self.__name = name
            else:
                self.__name = self.__mF.getName()
            self.__output = output
            self.__mF.setSupport([max(self.__mF.getSupport()[0], self.__output.getDomain()[0]),min(self.__mF.getSupport()[1], self.__output.getDomain()[1])])
            self.__centroid = IT2Engine_Centroid(100).getCentroid(m)
        else:
            self.__centroid = centroid

    def getName(self): return self.__name

    def getOutput(self): return self.__output

    def setName(self, name): self.__name = name

    def setOutput(self, output:Output): self.__output = output

    def getMembershipFunction(self): return self.__mF

    def getCentroid(self): return self.__centroid

    def renew(self, params):
        self.__mF.renewparams(params)
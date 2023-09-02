from common.Output import Output
from intervaltype2.system.IT2_Consequent import IT2_Consequent


class T2_Consequent:
    __cacheTypeReducedCentroid = False

    def __init__(self, name: str, set, output:Output):
        self.__set = set
        self.__name = name
        self.__output = output

        set.setSupport([max(set.getSupport()[0], self.__output.getDomain()[0]), min(set.getSupport()[1], self.__output.getDomain()[1])])

    def getSet(self): return self.__set

    def getName(self): return self.__name

    def setName(self, name): self.__name = name

    def getOutput(self): return self.__output

    def setOutput(self, output: Output): self.__output = output

    def getConsequentsIT2Sets(self):
        cons = []
        for i in range(self.getSet().getNumberOfSlices()):
            cons.append(IT2_Consequent(self.__name+"_zSlices_"+str(i),self.getSet().getZSlice(i), self.getOutput(), None))

        return cons

    def equals(self, consequent):
        if self == consequent:
            return True
        if not isinstance(consequent, T2_Consequent):
            return False

        return self.getSet() == consequent.getSet()

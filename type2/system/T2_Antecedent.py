from datashape import Null

from common.Input import Input
from intervaltype2.system.IT2_Antecedent import IT2_Antecedent


class T2_Antecedent:
    __input: Input
    __name: str

    def __init__(self, name, set, input):
        self.__name = name
        self.__input = input
        self.__set = set

    def getInput(self): return self.__input

    def getName(self): return self.__name

    def setName(self, name:str): self.__name = name

    def getFS(self):
        if self.__set.getSupport()[0] <= self.__input.getInput() <= self.__set.getSupport()[0]:
            return self.__set.getFS(self.__input.getInput())
        else:
            return Null

    def getSet(self): return self.__set

    def getAntecedentasIT2Sets(self):
        ants = []
        for i in range(self.getSet().getNumberOfSlices()):
            ants.append(IT2_Antecedent(self.getName()+"_zSlice_"+str(i), self.getSet().getZSlice(i), self.getInput()))
        return ants

    def equals(self, antecedent):
        if self == antecedent:
            return True
        if not isinstance(antecedent, T2_Antecedent):
            return False

        return self.getSet() == antecedent.getSet()
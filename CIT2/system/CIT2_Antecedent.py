from CIT2.CIT2 import CIT2
from CIT2.Gen_CIT2 import Gen_CIT2
from common.Input import Input


class CIT2_Antecedent:

    # __set: Gen_CIT2
    __input: Input

    def __init__(self, set, input):
        self.__set = set
        self.__input = input
        self.__set.setSupport([max(self.__set.getSupport()[0], input.getDomain()[0]), min(self.__set.getSupport()[1], input.getDomain()[1])])

    def getCIT2(self): return self.__set

    def getInput(self): return self.__input

    def clone(self):
        new_input = Input(self.__input.getName(), self.__input.getDomain())
        new_input.setInput(self.__input.getInput())
        return CIT2_Antecedent(self.__set, new_input)

    def renewSet(self, intervals):
        self.__set.setIntervals(intervals)

    def renew(self, set):
        self.__init__(set, self.__input)
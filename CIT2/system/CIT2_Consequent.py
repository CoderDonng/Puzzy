from CIT2.operation.TupleOperation import TupleOperations
from common.Output import Output


class CIT2_Consequent:
    __output: Output

    def __init__(self, mf, output: Output):
        self.__mf = mf
        self.__output = output
        self.__mf.setSupport(TupleOperations().intersection(self.__mf.getSupport(), output.getDomain()))

    def clone(self):
        new_output = Output(self.__output.getName(), self.__output.getDomain())
        new_output.setDomain(self.__output.getDomain())
        return CIT2_Consequent(self.__mf, new_output)

    def getCIT2(self): return self.__mf

    def getOutput(self): return self.__output

    def getMF(self): return self.__mf

    def renewSet(self, intervals):
        self.__mf.setIntervals(intervals)

    def getCentroid(self):
        return self.__mf.getCentroid()

    def renew(self, set):
        self.__init__(set, self.__output)
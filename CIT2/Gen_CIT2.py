import random
from datashape import Null
from CIT2.Generator import Generator_Triangular, Generator_Gauangle, Generator_Trapezoidal
from CIT2.operation.TupleOperation import TupleOperations
from intervaltype2.IT2MF import IT2MF_Prototype
from type1.T1MF import T1MF_Triangular, T1MF_Prototype, T1MF_Gauangle, T1MF_Trapezoidal, T1MF_Gaussian


class Gen_CIT2(IT2MF_Prototype):

    def __init__(self, name, generator_type, intervals):
        super().__init__(name, None, None)
        self.generator_type = generator_type
        self.__intervals = intervals
        if generator_type == "T1MF_Triangular":
            if len(self.__intervals) != 3:
                raise Exception("please set different intervals for the 3 ends of T1MF_Triangular")
            self.__leftmostAES = T1MF_Triangular(self.getName()+"_leftmostAES",self.__intervals[0][0], self.__intervals[1][0], self.__intervals[2][0])
            self.__rightmostAES = T1MF_Triangular(self.getName()+"_rightmostAES",self.__intervals[0][1], self.__intervals[1][1], self.__intervals[2][1])

        elif generator_type == "T1MF_Gaussian":
            if len(self.__intervals) != 2:
                raise Exception("please set different intervals for the 3 ends of T1MF_Gaussian")
            self.__leftmostAES = T1MF_Gaussian(self.getName() + "_leftmostAES", self.__intervals[0][0], self.__intervals[1])
            self.__rightmostAES = T1MF_Gaussian(self.getName() + "_leftmostAES", self.__intervals[0][1], self.__intervals[1])

        elif generator_type == "T1MF_Gauangle":
            if len(self.__intervals) != 3:
                raise Exception("please set different intervals for the 3 ends of T1MF_Gauangle")
            self.__leftmostAES = T1MF_Gauangle(self.getName() + "_leftmostAES", self.__intervals[0][0], self.__intervals[1][0],
                                                 self.__intervals[2][0])
            self.__rightmostAES = T1MF_Gauangle(self.getName() + "_leftmostAES", self.__intervals[0][1], self.__intervals[1][1],
                                                  self.__intervals[2][1])

        elif generator_type == "T1MF_Trapezoidal":
            if len(self.__intervals) != 4:
                raise Exception("please set different intervals for the 3 ends of T1MF_Trapezoidal")
            self.__leftmostAES = T1MF_Trapezoidal(self.getName() + "_leftmostAES", [self.__intervals[0][0], self.__intervals[1][0],
                                               self.__intervals[2][0], self.__intervals[3][0]], Null)
            self.__rightmostAES = T1MF_Trapezoidal(self.getName() + "_leftmostAES", [self.__intervals[0][1], self.__intervals[1][1],
                                                self.__intervals[2][1]], Null)
        self.initializeUpperbound()
        self.initializeLowerbound()
        self.support = [self.__leftmostAES.getSupport()[0], self.__rightmostAES.getSupport()[1]]

    def setSupport(self, support):
        self.support = support
        self.__rightmostAES.setSupport(TupleOperations().intersection(support, self.__rightmostAES.getSupport()))
        self.__leftmostAES.setSupport(TupleOperations().intersection(support, self.__leftmostAES.getSupport()))
        if self.uMF is not None:
            self.uMF.setSupport(TupleOperations().intersection(support, self.uMF.getSupport()))
        if self.lMF is not None:
            self.lMF.setSupport(TupleOperations().intersection(support, self.lMF.getSupport()))

    def getLeftmostAES(self): return self.__leftmostAES

    def getRightmostAES(self): return self.__rightmostAES

    def getCentroid(self):
        l = 0
        r = 0
        for it in iter(self.__intervals):
            l += it[0]
            r += it[1]

        return (l+r)/(len(self.__intervals)*2)

    def initializeUpperbound(self):
        self.uMF = Boundary(self.name, "UPPERBOUND", self.__leftmostAES, self.__rightmostAES)

    def initializeLowerbound(self):
        self.lMF = Boundary(self.name, "LOWERBOUND", self.__leftmostAES, self.__rightmostAES)

    def toIT2(self):
        return IT2MF_Prototype("IT2_from_"+self.name, self.uMF, self.lMF)

    def getRandomAES(self):
        params = []
        for it in iter(self.__intervals):
            randomValue = random.uniform(it[0], it[1])
            params.append(randomValue)

        if self.generator_type == "T1MF_Triangular":
            return Generator_Triangular("randomAES",params[0], params[1], params[2])
        elif self.generator_type == "T1MF_Gauangle":
            return Generator_Gauangle("randomAES", params[0], params[1], params[2])
        elif self.generator_type == "T1MF_Trapezoidal":
            return Generator_Trapezoidal("randomAES", params, Null)

    def setIntervals(self, intervals):
        self.__init__(self.name, self.generator_type, intervals)

class Boundary(T1MF_Prototype):

    def __init__(self, name, type, leftmost_aes, rightmost_aes):
        super().__init__(name)
        self.__type = type
        self.__leftmostAES = leftmost_aes
        self.__rightmostAES = rightmost_aes
        self.support = [leftmost_aes.getSupport()[0], rightmost_aes.getSupport()[1]]

    def getFS(self, x):
        if self.__type == "UPPERBOUND":
            if type(self.__leftmostAES) is T1MF_Triangular:
                if self.__leftmostAES.getStart() <= x < self.__leftmostAES.getPeak():
                    return self.__leftmostAES.getFS(x)
                elif self.__leftmostAES.getPeak() <= x < self.__rightmostAES.getPeak():
                    return 1
                elif self.__rightmostAES.getPeak() <= x <= self.__rightmostAES.getEnd():
                    return self.__rightmostAES.getFS(x)
                else:
                    return 0
            elif type(self.__leftmostAES) is T1MF_Gauangle:
                if self.__leftmostAES.getStart() <= x < self.__leftmostAES.getPeak():
                    return self.__leftmostAES.getFS(x)
                elif self.__leftmostAES.getPeak() <= x < self.__rightmostAES.getPeak():
                    return 1
                elif self.__rightmostAES.getPeak() <= x <= self.__rightmostAES.getEnd():
                    return self.__rightmostAES.getFS(x)
                else:
                    return 0
            elif type(self.__leftmostAES) is T1MF_Trapezoidal:
                if self.__leftmostAES.getA() <= x < self.__leftmostAES.getB():
                    return self.__leftmostAES.getFS(x)
                elif self.__leftmostAES.getB() <= x < self.__rightmostAES.getC():
                    return 1
                elif self.__rightmostAES.getC() <= x <= self.__rightmostAES.getD():
                    return self.__rightmostAES.getFS(x)
                else:
                    return 0
            elif type(self.__leftmostAES) is T1MF_Gaussian:
                if self.__leftmostAES.getMean() <= x < self.__rightmostAES.getMean():
                    return 1
                elif self.getSupport()[0] <= x < self.__leftmostAES.getMean() or self.__rightmostAES.getMean() <= x <= self.getSupport()[1]:
                    return max(self.__leftmostAES.getFS(x), self.__rightmostAES.getFS(x))
                else:
                    return 0
        else:
            if type(self.__leftmostAES) is T1MF_Triangular:
                if self.__rightmostAES.getStart() <= x < self.__leftmostAES.getEnd():
                    return min(self.__leftmostAES.getFS(x), self.__rightmostAES.getFS(x))
                else:
                    return 0
            elif type(self.__leftmostAES) is T1MF_Gauangle:
                if self.__rightmostAES.getStart() <= x < self.__leftmostAES.getEnd():
                    return min(self.__leftmostAES.getFS(x), self.__rightmostAES.getFS(x))
                else:
                    return 0
            elif type(self.__leftmostAES) is T1MF_Trapezoidal:

                if self.__rightmostAES.getA() <= x <= self.__leftmostAES.getD():
                    return min(self.__leftmostAES.getFS(x), self.__rightmostAES.getFS(x))
                else:
                    return 0
            elif type(self.__leftmostAES) is T1MF_Gaussian:
                if self.getSupport()[0] <= x <= self.getSupport()[1]:
                    return min(self.__leftmostAES.getFS(x), self.__rightmostAES.getFS(x))
                else:
                    return 0

    def clone(self):
        return Boundary(self.name, self.__type, self.__leftmostAES, self.__rightmostAES)

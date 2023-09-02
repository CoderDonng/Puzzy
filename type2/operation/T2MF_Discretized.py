import numpy

from type1.operation.T1MF_Discretized import T1MF_Discretized
from type2.T2MF import T2MF_Prototype


class T2MF_Discretized(T2MF_Prototype):
    __set: [[]]
    __xDiscretizationValues = []
    __yDiscretizationValues = []
    __precision = 0.000001
    __xDiscretizationLevel: int
    __vSlices: list

    def __init__(self, t2set: T2MF_Prototype, primaryDiscretizationLevel: int, secondaryDiscretizationLevel: int):
        super().__init__("Discretized version of " + t2set.getName())
        self.support = t2set.getSupport()
        self.__xDiscretizationLevel = primaryDiscretizationLevel
        self.__xDiscretizationValues = numpy.zeros(primaryDiscretizationLevel).tolist()
        xStep: float = t2set.getSupport()[0]
        primStepSize: float = (t2set.getSupport()[1] - t2set.getSupport()[0]) / (primaryDiscretizationLevel - 1)

        if secondaryDiscretizationLevel is None:
            self.__vSlices: [T1MF_Discretized] = []
            for i in range(primaryDiscretizationLevel):
                self.__xDiscretizationValues[i] = xStep
                self.__vSlices.append(t2set.getFS(xStep))

                xStep += primStepSize
        else:
            secStepsize = 1.0 / (secondaryDiscretizationLevel - 1)
            self.__yDiscretizationValues = numpy.zeros(secondaryDiscretizationLevel).tolist()
            self.__set = []
            for i in range(primaryDiscretizationLevel):
                self.__set.append([])
                for j in range(secondaryDiscretizationLevel):
                    self.__set[i].append(0)

            for i in range(primaryDiscretizationLevel):
                yStep = 0
                self.__xDiscretizationValues[i] = xStep
                self.__vSlices.append(t2set.getFS(xStep))
                for j in range(secondaryDiscretizationLevel):
                    self.__yDiscretizationValues[j] = yStep
                    self.__set[i][j] = (t2set.getFS(xStep)).getFS(yStep)
                    yStep += secStepsize

                xStep += primStepSize

    def getPrimaryDiscretizationLevel(self):
        return self.__xDiscretizationLevel

    def getSetDataAt(self, xPoint: int, yPoint: int):
        if self.__set[xPoint][yPoint] > self.__precision:
            return self.__set[xPoint][yPoint]
        else:
            return 0

    def getDiscX(self, xPoint: int):
        return self.__xDiscretizationValues[xPoint]

    def getDiscY(self, yPoint: int):
        return self.__yDiscretizationValues[yPoint]

    def getSecondaryDiscretizationLevel(self):
        return len(self.__yDiscretizationValues)

    def getPrimaryDiscretizationValues(self):
        return self.__xDiscretizationValues

    def getSecondaryDiscretizationValues(self):
        return self.__yDiscretizationValues


from type1.T1MF import T1MF_Prototype


class T1MF_Union(T1MF_Prototype):

    def __init__(self, setA, setB):
        super().__init__("Union: " + setA.getName() + "_" + setB.getName())
        self.support = [min(setA.getSupport()[0], setB.getSupport()[0]), max(setA.getSupport()[1], setB.getSupport()[1])]
        self.__setA = setA
        self.__setB = setB

    def getFS(self, x: float):
        return max(self.__setA.getFS(x), self.__setB.getFS(x))

    def getAlphaCut(self, alpha: float):
        raise Exception("Not supported yet.")

    def getPeak(self):
        raise Exception("Not supported yet.")

    def getDefuzzifiedCentroid(self, numberOfDiscretizations: int):

        stepSize = (self.getSupport()[1] - self.getSupport()[0]) / (numberOfDiscretizations - 1)
        currentStep = self.getSupport()[0]
        numerator = 0.0
        denominator = 0.0
        fs = 0
        for i in range(numberOfDiscretizations):
            fs = self.getFS(currentStep)
            numerator += currentStep * fs
            denominator += fs
            currentStep += stepSize

        if denominator == 0.0:
            return 0.0
        else:
            return numerator / denominator
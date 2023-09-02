import math
from numpy import Inf
from type1.T1MF import T1MF_Prototype


class T1MF_Discretized(T1MF_Prototype):
    Set: list
    peak: float
    sorted: bool = False
    discretizationLevel: int
    __leftShoulder = False; __rightShoulder = False
    __leftShoulderStart = 0.0; __rightShoulderStart = 0.0
    __a:float;__temp2:float;__temp2:float;__temp3:float;__left:float; __right:float;__min:float
    __alphaCutDiscLevel = 60
    __alphaCutPrcsisionLimit = 0.01

    def __init__(self,name:str):
        super().__init__(name)
        self.Set = []
        self.support = [float, float]

    def addPoint(self,p):
        self.Set.append(p)
        self.sorted = False

    def addPoints(self,ps:list):
        for i in iter(ps):
            self.Set.append(i)
        self.sorted = False

    def getAlphaCutDisretizationLevel(self):
        return self.__alphaCutDiscLevel

    def setAlphaCutDisretizationLevel(self, alphaCutDiscLevel:int):
        self.__alphaCutDiscLevel = alphaCutDiscLevel

    def getNumberOfPoints(self):
        return len(self.Set)

    def getFS(self, x: float):
        if len(self.Set) == 0:
            return -1
        if self.__leftShoulder:
            if x < self.__leftShoulderStart:
                return 1
        if self.__rightShoulder:
            if x > self.__rightShoulderStart:
                return 1
        if x < self.getSupport()[0] or x > self.getSupport()[1]:
            return 0.0

        self.sort()
        for i in range(len(self.Set)):
            if self.Set[i][1] > x:
                return self.interpolate(i-1, x, i)
            elif self.Set[i][1] == x:
                return self.Set[i][0]

        return float(math.nan)

    def getAlphaCut(self, alpha: float):
        self.__left:float = 0.0; self.__right:float = 0.0

        if alpha == 0.0:
            return self.getSupport()

        if alpha == 1.0:
            for i in range(len(self.Set)):
                if self.Set[i][0] == 1.0:
                    self.__left = self.Set[i][1]

            for i in reversed(range(len(self.Set))):
                if self.Set[i][0] == 1.0:
                    self.__right = self.Set[i][1]

            return [self.__left, self.__right]

        # for other alphas between 0 and 1
        stepSize = (self.getSupport()[1]-self.getSupport()[0])/(self.__alphaCutDiscLevel-1)
        currentStep = self.getSupport()[0]

        for i in range(self.__alphaCutDiscLevel):
            temp = self.getFS(currentStep) - alpha
            if temp >= 0.0:
                self.__left = currentStep
                break

            currentStep += stepSize

        currentStep = self.getSupport()[1]

        for i in range(self.__alphaCutDiscLevel):
            temp = self.getFS(currentStep) - alpha
            if temp >= 0.0:
                self.__left = currentStep
                break
            currentStep -= stepSize

        alphaCut = [self.__left, self.__right]

        if abs(self.__left-self.__right)<self.__alphaCutPrcsisionLimit:
            alphaCut[1] = self.__left

        return alphaCut

    # Calcuates f(s) for input x through interpolation.
    def interpolate(self, x_0:int, x_1:float, x_2:int):
        a = (self.Set[x_2][1] - self.Set[x_0][1])/(x_1 - self.Set[x_0][1])
        return self.Set[x_0][0] - (self.Set[x_0][0] - self.Set[x_2][0])/a

    def getPointAt(self,i: int):
        self.sort()
        return self.Set[i]

    def getPeak(self):
        self.sort()
        # two x's if the set has a flat top
        xCoordinateofPeak = 0.0
        yValueAtCurrentPeak = 0.0
        secondX=0.0
        yValueAtCurrentPeak = self.getPointAt(0)[0]
        xCoordinateofPeak = self.getPointAt(0)[1]
        i = 1
        while i < len(self.Set):
            if self.getPointAt(i)[0] > yValueAtCurrentPeak:
                yValueAtCurrentPeak = self.getPointAt(i)[0]
                xCoordinateofPeak = self.getPointAt(i)[1]
            else:
                if self.getPointAt(i)[0] == yValueAtCurrentPeak:
                    while self.getPointAt(i)[0]== yValueAtCurrentPeak:
                        secondX = self.getPointAt(i)[1]
                        i+=1
                    return (xCoordinateofPeak+secondX)/2.0
                break
            i+=1

        return xCoordinateofPeak

    def getSupport(self):
        if len(self.Set) == 0:
            return None

        self.sort()
        if self.__leftShoulder:
            self.support = [float(-Inf), self.Set[len(self.Set)-1][1]]
        elif self.__rightShoulder:
            self.support = [self.Set[0][1], float(Inf)]
        else:
            self.support = [self.Set[0][1], self.Set[len(self.Set)-1][1]]

        return self.support

    def sort(self):
        if ~self.sorted and len(self.Set) != 0:
            # sort
            self.Set.sort()
            self.support[0] = self.Set[0][1]
            self.support[1] = self.Set[len(self.Set)-1][1]
            self.sorted = True
            # prune
            if len(self.Set)>1:
                lastX = self.Set[0][1]
                j=1
                while j<len(self.Set):
                    if self.Set[j][1] == lastX:
                        self.Set[j-1][0] = max(self.Set[j-1][0], self.Set[j][0])
                        j-=1
                    else:
                        lastX = self.Set[j][1]
                    j+=1

    def setLeftShoulderSet(self, shoulderStart:float):
        self.__leftShoulder = True
        self.__leftShoulderStart = shoulderStart
        self.support[0] = float(-Inf)

    def setRightShoulderSet(self, shoulderStart:float):
        self.__rightShoulder = True
        self.__rightShoulderStart = shoulderStart
        self.support[1] = float(Inf)

    def getPoints(self):
        self.sort()
        return self.Set

    def getDefuzzifiedCentroid(self, numberOfDiscretizations:int):
        stepSize = (self.getSupport()[0]+self.getSupport()[1])/(2*(numberOfDiscretizations-1))
        currentStep = self.getSupport()[0]
        numerator = 0.0
        denominator = 0.0
        fs = 0.0
        for item in iter(self.getPoints()):
            numerator += item[1]*item[0]
            denominator += item[0]

        if denominator == 0.0:
            return 0.0
        else:
            return numerator/denominator

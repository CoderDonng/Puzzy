import math

from datashape import Null
from numpy import Inf


class T1MF_Prototype():
    isLeftShoulder: bool = False
    isRightShoulder: bool = False
    name: str
    support: [float, float]

    def __init__(self, name):
        self.name = name
        self.support = None
        self.isRightShoulder = False
        self.isRightShoulder = False
        self.__DEBUG = False

    def getSupport(self):
        return self.support

    def setSupport(self, support):
        self.support = support

    def getName(self):
        return self.name

    def setName(self, name: str):
        self.name = name

    def isleftShoulder(self):
        return self.isLeftShoulder

    def isrightShoulder(self):
        return self.isRightShoulder

    def setleftShoulder(self, value: bool):
        self.isLeftShoulder = value

    def setrightShoulder(self, value: bool):
        self.isRightShoulder = value

    def getAlphaCut(self, param):
        pass




class T1MF_Singleton(T1MF_Prototype):
    __value: float

    def __init__(self, name, value):
        super().__init__(name)
        # self.name = name
        self.__value = value
        self.support = [value, value]

    def getValue(self):
        return self.__value

    def getFS(self, x: float):
        if x == self.__value:
            return 1.0
        else:
            return 0.0

    def getPeak(self):
        return self.getValue()

    def getAlphaCut(self, alpha: float):
        return [self.__value, self.__value]


class T1MF_Triangular(T1MF_Prototype):
    __start: float
    __peak: float
    __end: float
    __output: float

    def __init__(self, name: str, start, peak, end):
        super().__init__(name)
        self.__start = start
        self.__peak = peak
        self.__end = end
        self.support = [start, end]

    def getFS(self, x: float):
        if self.isLeftShoulder and x <= self.__peak:
            return 1.0
        if self.isRightShoulder and x >= self.__peak:
            return 1.0

        if self.__start < x < self.__peak:
            self.__output = (x - self.__start) / (self.__peak - self.__start)
        elif x == self.__peak:
            self.__output = 1.0
        elif self.__peak < x < self.__end:
            self.__output = (self.__end - x) / (self.__end - self.__peak)
        else:
            self.__output = 0.0

        return self.__output

    def getStart(self):
        return self.__start

    def getPeak(self):
        return self.__peak

    def getEnd(self):
        return self.__end

    def setStart(self, start):
        self.__start = start

    def setPeak(self, peak):
        self.__peak = peak

    def setEnd(self, end):
        self.__end = end


class T1MF_Gaussian(T1MF_Prototype):
    __mean: float
    __spread: float

    def __init__(self, name: str, mean: float, spread: float):
        super().__init__(name)
        self.__mean = mean
        self.__spread = spread
        self.support = [mean - 4 * spread, mean + 4 * spread]

    def getFS(self, x: float):
        if self.getSupport()[0] <= x <= self.getSupport()[1]:
            if self.isLeftShoulder and x <= self.__mean:
                return 1.0
            if self.isRightShoulder and x >= self.__mean:
                return 1.0
            return math.exp(-0.5 * math.pow((x - self.__mean) / self.__spread, 2))
        else:
            return 0.0

    def getSpread(self):
        return self.__spread

    def getMean(self):
        return self.__mean

    def getPeak(self):
        return self.getMean()


class T1MF_Gauangle(T1MF_Prototype):
    __spreadForLeft: float
    __spreadForRight: float
    __start: float
    __center: float
    __end: float
    __transitionPointLeft: float
    __transitionPointRight: float
    __leftCalculationPoint: float
    __rightCalculationPoint: float
    __similarToGaussian = 0.5

    def __init__(self, name, start, center, end):
        super().__init__(name)
        self.__center = center
        self.__start = start
        self.__end = end

        if start == center:
            self.isLeftShoulder = True
        if end == center:
            self.isRightShoulder = True

        self.__spreadForLeft = (center - start) * (1.0 - self.__similarToGaussian)
        self.__spreadForRight = (end - center) * (1.0 - self.__similarToGaussian)
        self.support = [start, end]
        self.__transitionPointLeft = center - (center - start) * self.__similarToGaussian
        if self.__spreadForLeft == 0 or math.isnan(self.__spreadForLeft):
            ab = self.getLineEquationParameters([start, 0.0], [self.__transitionPointLeft, math.exp(
                -0.5 * math.pow(math.nan, 2))])
        else:
            ab = self.getLineEquationParameters([start, 0.0], [self.__transitionPointLeft, math.exp(
                -0.5 * math.pow((self.__transitionPointLeft - center) / self.__spreadForLeft, 2))])
        self.__leftCalculationPoint = self.getXForYOnLine(1.0, ab)

        self.__transitionPointRight = center + ((end - center) * self.__similarToGaussian)
        if self.__spreadForRight == 0 or math.isnan(self.__spreadForRight):
            ab_2 = self.getLineEquationParameters([self.__transitionPointRight, math.exp(
                -0.5 * math.pow(math.nan, 2))], [end, 0.0])
        else:
            ab_2 = self.getLineEquationParameters([self.__transitionPointRight, math.exp(
                -0.5 * math.pow((self.__transitionPointRight - center) / self.__spreadForRight, 2))], [end, 0.0])
        self.__rightCalculationPoint = self.getXForYOnLine(1.0, ab_2)

    def getFS(self, x: float):
        if self.support[0] <= x <= self.support[1]:
            if self.isLeftShoulder and x <= self.__center:
                return 1.0
            if self.isRightShoulder and x >= self.__center:
                return 1.0

            if x <= self.__transitionPointLeft:
                return (x - self.__start) / (self.__leftCalculationPoint - self.__start)
            elif x <= self.__transitionPointRight:
                if x <= self.__center:
                    return math.exp(-0.5 * math.pow(((x - self.__center) / self.__spreadForLeft), 2))
                else:
                    return math.exp(-0.5 * math.pow(((x - self.__center) / self.__spreadForRight), 2))
            else:
                return (self.__end - x) / (self.__end - self.__rightCalculationPoint)
        else:
            return 0.0

    def getPeak(self):
        return self.getMean()

    def getMean(self):
        return self.__center

    def getStart(self):
        return self.__start

    def getEnd(self):
        return self.__end

    def getLineEquationParameters(self, x: [float, float], y: [float, float]):

        ab = [0, 0]
        if y[0] - x[0] == 0:
            ab[0] = math.nan
        else:
            ab[0] = (y[1] - x[1]) / (y[0] - x[0])
        ab[1] = x[1] - ab[0] * x[0]

        return ab

    def getXForYOnLine(self, y: float, ab: list):
        return (y - ab[1]) / ab[0]


class T1MF_Trapezoidal(T1MF_Prototype):
    __a: float
    __b: float
    __c: float
    __d: float
    __lS: float = float(math.nan)
    __rS: float = float(math.nan)
    __lI: float = float(math.nan)
    __rI: float = float(math.nan)
    __output: float
    __peak: float
    __ylevels = [1.0, 1.0]

    def __init__(self, name: str, parameters: list, yLevels):
        super().__init__(name)
        self.__a = parameters[0]
        self.__b = parameters[1]
        self.__c = parameters[2]
        self.__d = parameters[3]
        self.support = [self.__a, self.__d]
        if yLevels != Null:
            self.__ylevels[0] = yLevels[0]
            self.__ylevels[1] = yLevels[1]

    def getFS(self, x: float):

        if self.isLeftShoulder and x <= self.__c:
            return 1.0
        if self.isRightShoulder and x >= self.__b:
            return 1.0

        if self.__b > x > self.__a:
            self.__output = self.__ylevels[0] * (x - self.__a) / (self.__b - self.__a)
        elif self.__b <= x <= self.__c:
            if self.__ylevels[0] == self.__ylevels[1]:
                self.__output = self.__ylevels[0]
            elif self.__ylevels[0] < self.__ylevels[1]:
                self.__output = (self.__ylevels[1] * x - self.__ylevels[0] * x - self.__ylevels[1] * self.__b +
                                 self.__ylevels[0] * self.__b) / (self.__c - self.__b) + self.__ylevels[0]
            else:
                self.__output = (self.__ylevels[1] * x - self.__ylevels[0] * x - self.__ylevels[1] * self.__b +
                                 self.__ylevels[0] * self.__b) / (self.__c - self.__b) + self.__ylevels[0]
            if self.__output < 0: self.__output = 0
        elif self.__c < x < self.__d:
            self.__output = self.__ylevels[1] * (self.__d - x) / (self.__d - self.__c)
        else:
            self.__output = 0

        if abs(1 - self.__output) < 0.000001: self.__output = 1.0
        if abs(self.__output) < 0.000001: self.__output = 0.0

        return self.__output

    def getA(self):
        return self.__a

    def getB(self):
        return self.__b

    def getC(self):
        return self.__c

    def getD(self):
        return self.__d

    def getParameters(self):
        return [self.__a, self.__b, self.__c, self.__d]

    def getPeak(self):
        if math.isnan(self.__peak):
            self.__peak = (self.__b + self.__c) / 2.0
        return self.__peak

    def setPeak(self, peak: float):
        self.__peak = peak

    def getylevels(self):
        return self.__ylevels

    def setylevels(self, ylevels: list):
        self.__ylevels = ylevels


class T1MF_Cylinder(T1MF_Prototype):
    __membershipDegree: float

    def __init__(self, name: str, membershipDegree: float):
        super().__init__(name)
        self.__membershipDegree = membershipDegree
        if self.__membershipDegree < 0 or self.__membershipDegree > 1.0:
            raise Exception("The membership degree should be between 0 and 1.")

        self.support = [float(-Inf), float(Inf)]

    def getFS(self, x: float):
        return self.__membershipDegree

    def getAlphaCut(self, alpha: float):
        if alpha <= self.__membershipDegree:
            return [float(-Inf), float(Inf)]
        else:
            return None


class T1MF_IFRS(T1MF_Prototype):
    __firingStrength: float

    def __init__(self, name, consequent, firing_strength: float):
        super().__init__(name)
        self.__firingStrength = firing_strength
        self.__consequent = consequent
        self.support = consequent.getSupport()

    def clone(self):
        return T1MF_IFRS(self.name, self.__consequent, self.__firingStrength)

    def getPeak(self):
        raise Exception("Not supported yet.")

    def getFS(self, x):
        if self.support[0] <= x <= self.support[1]:
            a = self.__consequent.getFS(x)
            return min(a, self.__firingStrength)
        return 0

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


class T1MF_Negate(T1MF_Prototype):

    def __init__(self, name, to_negate, support):
        super().__init__(name)
        self.toNegate = to_negate
        self.support = support

    def getPeak(self): raise Exception("Not supported yet.")

    def getFS(self, x):
        if self.support[0] <= x <= self.support[1]:
            return 1 - self.toNegate.getFS(x)
        return 0

    def clone(self): return T1MF_Negate(self.name, self.toNegate, self.support)


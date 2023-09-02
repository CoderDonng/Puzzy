import math
from type1.T1MF import T1MF_Cylinder, T1MF_Gaussian, T1MF_Trapezoidal, T1MF_Triangular


class IT2MF_Prototype:
    name: str
    support: []
    isLeftShoulder = False
    isRightShoulder = False
    uMF = None
    lMF = None

    def __init__(self, name, uMF, lMF):
        self.name = name
        self.uMF = uMF
        self.lMF = lMF
        if uMF is not None and lMF is not None:
            self.support = [min(uMF.getSupport()[0], lMF.getSupport()[0]),
                            max(uMF.getSupport()[1], lMF.getSupport()[1])]
            self.uMF.setSupport(self.support)
            self.lMF.setSupport(self.support)
        else:
            self.support = None

    def setSupport(self, support):
        self.support = support

    def getSupport(self):
        return self.support

    def setName(self, name):
        self.name = name

    def getName(self):
        return self.name

    def getFSAverage(self, x):
        fs = self.getFS(x)
        return (fs[0] + fs[1]) / 2.0

    def getFS(self, x: float):
        return [self.lMF.getFS(x), self.uMF.getFS(x)]

    def isleftShoulder(self):
        return self.isLeftShoulder

    def isrightShoulder(self):
        return self.isLeftShoulder

    def setLeftShoulder(self, isLeftShoulder):
        self.isLeftShoulder = isLeftShoulder

    def setRightShoulder(self, isRightShoulder):
        self.isRightShoulder = isRightShoulder

    def getLowerBound(self, x: float):
        return self.lMF.getFS(x)

    def getUpperBound(self, x: float):
        return self.uMF.getFS(x)

    def getLMF(self):
        return self.lMF

    def getUMF(self):
        return self.uMF

    def getPeak(self):
        if self.uMF.getPeak() == self.lMF.getPeak():
            return self.uMF.getPeak()
        else:
            return (self.uMF.getPeak() + self.lMF.getPeak()) / 2.0

    def renewparams(self, params):
        pass


class IT2MF_Cylinder(IT2MF_Prototype):
    def __init__(self, name: str, primer: [float, float], uML, lML):
        super().__init__(name, None, None)
        if primer is None and uML is None and lML is None:
            raise Exception("IT2MF_Cylinder primer is NULL!")
        elif primer is not None and uML is None and lML is None:
            if primer[0] > primer[1]:
                if primer[0] - primer[1] < 0.000001:
                    primer[0] = primer[1]
                else:
                    raise Exception("Lower firing strength (" + str(
                        primer[0]) + ") should not be higher than Upper firing strength (" + str(primer[1]) + ").")

            self.uMF = T1MF_Cylinder(name + "_uMF", primer[1])
            self.lMF = T1MF_Cylinder(name + "_lMF", primer[0])
            self.support = [float(-math.inf), float(math.inf)]
        elif primer is None and uML is not None and lML is not None:
            super().__init__(name, uML, lML)
        else:
            raise Exception("can not creat IT2MF_Cylinder by this way")


class IT2MF_Gauangle(IT2MF_Prototype):
    __leftShoulder = False
    __rightShoulder = False

    def __init__(self, name, uMF, lMF):
        super().__init__(name, uMF, lMF)

    def toString(self):
        s = self.getName()+" - IT2 Gauangle with UMF:\n"+str(self.getUMF())+" and LMF:\n"+str(self.getLMF())
        if self.isLeftShoulder:
            s += "\n (LeftShoulder)"
        if self.isRightShoulder:
            s += "\n (RightShoulder)"
        return s

    def getFS(self, x: float):
        l = self.lMF.getFS(x)
        u = self.uMF.getFS(x)

        if self.lMF.getPeak() == self.uMF.getPeak():
            return [min(l, u), max(l, u)]
        else:
            if min(self.lMF.getPeak(), self.uMF.getPeak()) <= x <= max(self.lMF.getPeak(), self.uMF.getPeak()):
                return [min(l, u), 1.0]
            else:
                return [min(l, u), max(l, u)]


class IT2MF_Gaussian(IT2MF_Prototype):

    def __init__(self, name: str, uMF: T1MF_Gaussian, lMF: T1MF_Gaussian):
        super().__init__(name, uMF, lMF)
        if uMF is not None and lMF is not None:
            if uMF.getMean() < lMF.getMean():
                raise Exception(
                    "By convention, the mean of the upper membership function should be larger than that of the lower membership function.")
            if uMF.getSpread() < lMF.getSpread():
                raise Exception(
                    "By convention, the st. dev. (spread) of the upper membership function should be larger than that of the lower membership function.")
            self.support = uMF.getSupport()

    def getUMF(self):
        return self.uMF

    def getLMF(self):
        return self.lMF

    def getFS(self, x: float):
        if x < self.support[0]:
            return [0, 0]
        if x > self.support[1]:
            return [0, 0]

        if self.lMF.getMean() == self.uMF.getMean():
            return [math.exp(-0.5 * math.pow((x - self.lMF.getMean()) / self.lMF.getSpread(), 2)),
                    math.exp(-0.5 * math.pow((x - self.uMF.getMean()) / self.uMF.getSpread(), 2))]
        else:
            if x < self.lMF.getMean():
                temp = math.exp(-0.5 * math.pow((x - self.lMF.getMean()) / self.lMF.getSpread(), 2))
            elif x > self.uMF.getMean():
                temp = math.exp(-0.5 * math.pow((x - self.uMF.getMean()) / self.uMF.getSpread(), 2))
            else:
                temp = 1.0

            if x < (self.lMF.getMean() + self.uMF.getMean()) / 2:
                temp2 = math.exp(-0.5 * math.pow((x - self.uMF.getMean()) / self.uMF.getSpread(), 2))
            else:
                temp2 = math.exp(-0.5 * math.pow((x - self.lMF.getMean()) / self.lMF.getSpread(), 2))

            return [min(temp, temp2), max(temp, temp2)]


class IT2MF_Trapezoidal(IT2MF_Prototype):

    def __init__(self, name, upper: T1MF_Trapezoidal, lower: T1MF_Trapezoidal):
        super().__init__(name, upper, lower)
        if upper.getA() > lower.getA() or upper.getB() > lower.getB() or upper.getC() < lower.getC() or upper.getD() < lower.getD():
            raise Exception("The upper membership function needs to be higher than the lower membership function.")

        self.lMF = lower
        self.uMF = upper
        self.support = upper.getSupport()


class IT2MF_Triangular(IT2MF_Prototype):
    def __init__(self, name: str, uMF: T1MF_Triangular, lMF: T1MF_Triangular):
        super().__init__(name, uMF, lMF)
        if uMF.getStart() > lMF.getStart() or uMF.getEnd() < lMF.getEnd():
            raise Exception("The upper membership function needs to be higher than the lower membership function.")

    def getLMF(self): return self.lMF

    def getUMF(self): return self.uMF

    def getFS(self, x: float):
        l = self.lMF.getFS(x)
        u = self.uMF.getFS(x)

        if self.lMF.getPeak() == self.uMF.getPeak():
            return [min(l, u), max(l, u)]
        else:
            if min(self.lMF.getPeak(), self.uMF.getPeak()) <= x <= max(self.lMF.getPeak(), self.uMF.getPeak()):
                return [min(l, u), 1.0]
            else:
                return [min(l, u), max(l, u)]

    def compareTo(self, o):
        if ~isinstance(o, IT2MF_Triangular):
            raise Exception("A IT2MF_Triangular object is expected for comparison with another IntervalT2MF_Triangular object.")
        else:
            if self.getLMF().getStart() == o.getLMF().getStart() and self.getLMF().getPeak() == o.getLMF().getPeak() and self.getLMF().getEnd() == o.getLMF().getEnd() and self.getUMF().getStart() == o.getUMF().getStart() and self.getUMF().getPeak() == o.getUMF().getPeak() and self.getUMF().getEnd() == o.getUMF().getEnd():
                return 0
            elif self.getLMF().getStart() <= o.getLMF().getStart() and self.getLMF().getPeak() <= o.getLMF().getPeak() and self.getLMF().getEnd() <= o.getLMF().getEnd() and self.getUMF().getStart() <= o.getUMF().getStart() and self.getUMF().getPeak() <= o.getUMF().getPeak() and self.getUMF().getEnd() <= o.getUMF().getEnd():
                return -1
            else:
                return 1

    def renewparams(self, params):
        self.uMF.setStart(params[0])
        self.uMF.setPeak(params[2])
        self.uMF.setEnd(params[4])
        self.lMF.setStart(params[1])
        self.lMF.setPeak(params[2])
        self.lMF.setEnd(params[3])


class Gen_IT2MF(IT2MF_Prototype):

    def __init__(self, name, upperbound, lowerbound):
        super().__init__(name, upperbound, lowerbound)
        self.support = upperbound.getSupport()

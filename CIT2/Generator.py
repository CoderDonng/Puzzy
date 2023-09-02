import math
from datashape import Null
from type1.T1MF import T1MF_Prototype, T1MF_Gauangle, T1MF_Gaussian, T1MF_Trapezoidal, T1MF_Triangular


class Generator_Prototype(T1MF_Prototype):

    def __init__(self, name, T1MF):
        super().__init__(name)
        self.T1MF = T1MF
        self.typeCastT1MF()
        self.minPoints = self.computeMinPoints()
        self.maxPoints = self.computeMaxPoints()
        if self.T1MF.isleftShoulder():
            self.setleftShoulder(True)
        if self.T1MF.isrightShoulder():
            self.setrightShoulder(True)

    def getSupport(self): return self.T1MF.getSupport()

    def setSupport(self, support): self.T1MF.setSupport(support)

    def getPeak(self): return self.T1MF.getPeak()

    def getFS(self, x): return self.T1MF.getFS(x)

    def getMinPoints(self): return self.minPoints

    def getMaxPoints(self): return self.maxPoints

    def getAlphaCut(self, param): return self.T1MF.getAlphaCut(param)

    def compareTo(self, o: object): raise Exception("Not supported yet.")

    def setleftShoulder(self, value: bool):
        if value == self.isLeftShoulder:
            return
        self.T1MF.setleftShoulder(value)
        self.isLeftShoulder = value
        if value:
            self.maxPoints = self.computeLeftShoulderMaxPoints()
            self.minPoints = self.computeLeftShoulderMinPoints()
            self.isRightShoulder = False
            self.T1MF.setrightShoulder(False)
        else:
            self.maxPoints = self.computeMaxPoints()
            self.minPoints = self.computeMinPoints()

    def setrightShoulder(self, value: bool):
        if value == self.isRightShoulder:
            return
        self.T1MF.setrightShoulder(value)
        self.isRightShoulder = value
        if value:
            self.maxPoints = self.computeRightShoulderMaxPoints()
            self.minPoints = self.computeRightShoulderMinPoints()
            self.isLeftShoulder = False
            self.T1MF.setleftShoulder(False)
        else:
            self.maxPoints = self.computeMaxPoints()
            self.minPoints = self.computeMinPoints()

    def typeCastT1MF(self): pass

    def computeLeftShoulderMaxPoints(self): pass

    def computeRightShoulderMaxPoints(self): pass

    def computeLeftShoulderMinPoints(self): pass

    def computeRightShoulderMinPoints(self): pass

    def computeMaxPoints(self): pass

    def computeMinPoints(self): pass

    def clone(self): pass

    def setMinPoints(self, min_points): self.minPoints = min_points

    def setMaxPoints(self, max_points): self.maxPoints = max_points

    def setShiftedSupportSet(self, original: T1MF_Prototype, shifted):
        if original.isleftShoulder():
            shifted.setleftShoulder(True)
            shifted.setSupport([original.getSupport()[0], shifted.getSupport()[1]])

        if original.isrightShoulder():
            shifted.setrightShoulder(True)
            shifted.setSupport([shifted.getSupport()[0], original.getSupport()[1]])

class Generator_Gauangle(Generator_Prototype):
    __gauangle: T1MF_Gauangle

    def __init__(self, name, start, center, end):
        super().__init__(name, T1MF_Gauangle(name, start, center, end))

    def typeCastT1MF(self): self.__gauangle = self.T1MF

    def computeMinPoints(self):
        min_points = []
        min_points.append([self.__gauangle.getStart(), self.__gauangle.getStart()])
        min_points.append([self.__gauangle.getEnd(), self.__gauangle.getEnd()])
        return min_points #min_points:[[start, start], [end, end]]

    def computeMaxPoints(self):
        max_points = []
        max_points.append([self.__gauangle.getPeak(), self.__gauangle.getPeak()])
        return max_points #max_points:[[peak, peak]]

    def computeRightShoulderMinPoints(self):
        min_points = []
        min_points.append([self.__gauangle.getStart(), self.__gauangle.getStart()])
        return min_points #为何rightshoulderminpoints的值为[[start, start]]， rightshouldermaxpoints值为[[peak, inf]]

    def computeRightShoulderMaxPoints(self):
        max_points = []
        max_points.append([self.__gauangle.getPeak(), float(math.inf)])
        return max_points

    def computeLeftShoulderMinPoints(self):
        min_points = []
        min_points.append([self.__gauangle.getEnd(), self.__gauangle.getEnd()])
        return min_points

    def computeLeftShoulderMaxPoints(self):
        max_points = []
        max_points.append([float(-math.inf), self.__gauangle.getPeak()])
        return max_points

    def clone(self):
        clone = Generator_Gauangle(self.getName(), self.__gauangle.getStart(), self.__gauangle.getPeak(), self.__gauangle.getEnd())
        clone.setleftShoulder(self.isLeftShoulder)
        clone.setrightShoulder(self.isRightShoulder)
        clone.setSupport(self.support)
        return clone

    def shiftFunction(self, name, shifting_step):
        shifted_fun = Generator_Gauangle(name, self.__gauangle.getStart()+shifting_step, self.__gauangle.getPeak()+shifting_step, self.__gauangle.getEnd()+shifting_step)
        self.setShiftedSupportSet(self.__gauangle, shifted_fun)
        return shifted_fun


class Generator_Gaussian(Generator_Prototype):
    __gaussian: T1MF_Gaussian

    def __init__(self, name, mean, spread):
        super().__init__(name, T1MF_Gaussian(name, mean, spread))

    def typeCastT1MF(self): self.__gaussian = self.T1MF

    def computeMinPoints(self): return Null

    def computeMaxPoints(self):
        max_points = []
        max_points.append([self.__gaussian.getMean(), self.__gaussian.getMean()])
        return max_points

    def computeRightShoulderMinPoints(self): return Null

    def computeRightShoulderMaxPoints(self):
        max_points = []
        max_points.append([self.__gaussian.getMean(), float(math.inf)])
        return max_points

    def computeLeftShoulderMinPoints(self): return Null

    def computeLeftShoulderMaxPoints(self):
        max_points = []
        max_points.append([float(-math.inf), self.__gaussian.getMean()])
        return max_points

    def shiftFunction(self, name, shifting_factor):
        shifted_fun = Generator_Gaussian(name, self.__gaussian.getMean()+shifting_factor, self.__gaussian.getSpread())
        self.setShiftedSupportSet(self.__gaussian, shifted_fun)
        return shifted_fun

    def clone(self):
        clone = Generator_Gaussian(self.getName(), self.__gaussian.getMean(), self.__gaussian.getSpread())
        clone.setleftShoulder(self.isLeftShoulder)
        clone.setrightShoulder(self.isRightShoulder)
        clone.setSupport(self.support)
        return clone


class Generator_Negate(T1MF_Prototype):

    def __init__(self, name, to_negate):
        super().__init__(name)
        self.toNegate = to_negate
        self.support = to_negate.getSupport()

    def shiftFunction(self, name, value):
        return Generator_Negate(name, self.toNegate.shiftFuntion(name, value))

    def getMinPoints(self): return self.toNegate.getMaxPoints()

    def getMaxPoints(self): return self.toNegate.getMinPoints()

    def getPeak(self): raise Exception("Not supported yet.")

    def getAlphaCut(self, param): raise Exception("Not supported yet.")

    def getFS(self, x): return 1-self.toNegate.getFS(x)

    def compareTo(self, o:object): raise Exception("Not supported yet.")

    def clone(self): return Generator_Negate(self.name, self.toNegate)


class Generator_Trapezoidal(Generator_Prototype):
    __trapezoid: T1MF_Trapezoidal

    def __init__(self, name, params, ylevels):
        super().__init__(name, T1MF_Trapezoidal(name, params, ylevels))

    def typeCastT1MF(self): self.__trapezoid = self.T1MF

    def computeMinPoints(self):
        min_points = []
        min_points.append([self.__trapezoid.getA(), self.__trapezoid.getA()])
        min_points.append([self.__trapezoid.getD(), self.__trapezoid.getD()])
        return min_points

    def computeMaxPoints(self):
        max_points = []
        max_points.append([self.__trapezoid.getB(), self.__trapezoid.getC()])
        return max_points

    def computeRightShoulderMinPoints(self):
        min_points = []
        min_points.append([self.__trapezoid.getA(), self.__trapezoid.getA()])
        return min_points

    def computeRightShoulderMaxPoints(self):
        max_points = []
        max_points.append([self.__trapezoid.getB(), float(math.inf)])
        return max_points

    def computeLeftShoulderMinPoints(self):
        min_points = []
        min_points.append([self.__trapezoid.getD(), self.__trapezoid.getD()])
        return min_points

    def computeLeftShoulderMaxPoints(self):
        max_points = []
        max_points.append([float(-math.inf), self.__trapezoid.getC()])
        return max_points

    def shiftFunction(self, name, shifting_factor):
        shifted_fun = Generator_Trapezoidal(name, [self.__trapezoid.getA()+shifting_factor, self.__trapezoid.getB()+shifting_factor, self.__trapezoid.getC()+shifting_factor, self.__trapezoid.getD()+shifting_factor], self.__trapezoid.getylevels())
        self.setShiftedSupportSet(self.__trapezoid, shifted_fun)
        return shifted_fun

    def clone(self):
        clone = Generator_Trapezoidal(self.getName(), self.__trapezoid.getParameters(), self.__trapezoid.getylevels())
        clone.setleftShoulder(self.isLeftShoulder)
        clone.setrightShoulder(self.isRightShoulder)
        clone.setSupport(self.support)
        return clone


class Generator_Triangular(Generator_Prototype):
    __triangle: T1MF_Triangular

    def __init__(self, name, start, peak, end):
        super().__init__(name, T1MF_Triangular(name, start, peak, end))

    def typeCastT1MF(self):
        self.__triangle = self.T1MF

    def computeMinPoints(self):
        min_points = []
        min_points.append([self.__triangle.getStart(), self.__triangle.getStart()])
        min_points.append([self.__triangle.getEnd(), self.__triangle.getEnd()])
        return min_points

    def computeMaxPoints(self):
        max_points = []
        max_points.append([self.__triangle.getPeak(), self.__triangle.getPeak()])
        return max_points

    def computeRightShoulderMinPoints(self):
        min_points = []
        min_points.append([self.__triangle.getStart(), self.__triangle.getStart()])
        return min_points

    def computeRightShoulderMaxPoints(self):
        max_points = []
        max_points.append([self.__triangle.getPeak(), float(math.inf)])
        return max_points

    def computeLeftShoulderMinPoints(self):
        min_points = []
        min_points.append([self.__triangle.getEnd(), self.__triangle.getEnd()])
        return min_points

    def computeLeftShoulderMaxPoints(self):
        max_points = []
        max_points.append([float(-math.inf), self.__triangle.getPeak()])
        return max_points

    def shiftFunction(self, name, shifting_factor):
        shifted_fun = Generator_Triangular(name, self.__triangle.getStart() + shifting_factor,
                                                   self.__triangle.getPeak() + shifting_factor,
                                                   self.__triangle.getEnd() + shifting_factor)
        self.setShiftedSupportSet(self.__triangle, shifted_fun)
        return shifted_fun

    def clone(self):
        clone = Generator_Triangular(self.getName(), self.__triangle.getStart(), self.__triangle.getPeak(), self.__triangle.getEnd())
        clone.setleftShoulder(self.isLeftShoulder)
        clone.setrightShoulder(self.isRightShoulder)
        clone.setSupport(self.support)
        return clone


class Shifted_MF(T1MF_Prototype):
    __shiftingFactor: float
    __minPoints: []
    __maxPoints: []

    def __init__(self, mf_to_shift, shifting_factor):
        super().__init__(mf_to_shift.getName()+" shifted by "+str(shifting_factor))
        self.MFtoShift = mf_to_shift
        self.shiftingFactor = shifting_factor
        self.initializeMinMaxPoints()
        self.initializeSupportSet()
        self.isLeftShoulder = self.MFtoShift.isleftShoulder()
        self.isRightShoulder = self.MFtoShift.isrightShoulder()
        self.support = [self.MFtoShift.getSupport()[0]-shifting_factor, self.MFtoShift.getSupport()[1]-shifting_factor]

    def shiftFunction(self, a: str, b: str): return Null

    def setleftShoulder(self, value: bool):
        if value == self.isLeftShoulder:
            return
        self.MFtoShift.setleftShoulder(value)
        self.isLeftShoulder = value
        self.initializeSupportSet()

    def setrightShoulder(self, value: bool):
        if value == self.isRightShoulder:
            return
        self.MFtoShift.setrightShoulder(value)
        self.isRightShoulder = value
        self.initializeSupportSet()

    def initializeSupportSet(self):
        if self.MFtoShift.isleftShoulder() and self.shiftingFactor > 0:
            left = self.MFtoShift.getSupport()[0]
        else:
            left = self.MFtoShift.getSupport()[0]+self.shiftingFactor

        if self.MFtoShift.isrightShoulder() and self.shiftingFactor <0:
            right = self.MFtoShift.getSupport()[1]
        else:
            right = self.MFtoShift.getSupport()[1]+self.shiftingFactor

        self.support = [left, right]

    def initializeMinMaxPoints(self):
        self.__minPoints = self.shiftIntervals(self.MFtoShift.getMinPoints())
        self.__maxPoints = self.shiftIntervals(self.MFtoShift.getMaxPoints())

    def shiftIntervals(self, intervals):

        if intervals == Null:
            return Null

        result = []
        for it in iter(intervals):
            result.append([it[0]+self.shiftingFactor, it[1]+self.shiftingFactor])
        return result

    def getMinPoints(self): return self.__minPoints

    def getMaxPoints(self): return self.__maxPoints

    def getPeak(self): return self.MFtoShift.getPeak()+self.shiftingFactor

    def getAlphaCut(self, param): raise Exception("Not supported yet.")

    def compareTo(self, o: object):
        raise Exception("Not supported yet.")

    def clone(self):
        return Shifted_MF(self.MFtoShift, self.shiftingFactor)

    def getFS(self, x):
        mf_support = self.MFtoShift.getSupport()
        self.MFtoShift.setSupport(self.support)
        membership_degree = self.MFtoShift.getFS(x-self.shiftingFactor)
        self.MFtoShift.setSupport(mf_support)
        return membership_degree

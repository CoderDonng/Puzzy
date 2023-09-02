import math

import numpy
from datashape import Null

from intervaltype2.IT2MF import IT2MF_Cylinder, IT2MF_Gaussian, IT2MF_Trapezoidal, IT2MF_Triangular
from type1.T1MF import T1MF_Prototype, T1MF_Gaussian, T1MF_Trapezoidal, T1MF_Triangular
from type1.operation.T1MF_Discretized import T1MF_Discretized


class T2MF_Prototype:
    zSlices: []
    support: []
    name: str
    numberOfzLevels: int
    z_stepSize: float
    slices_zValues: [float]
    slices_fs: [float]
    isLeftShoulder = False
    isRightShoulder = False

    def __init__(self, name):
        self.name = name

    def setSupport(self, support):
        self.support = support

    def setName(self, name):
        self.name = name

    def getNumberOfSlices(self):
        return self.numberOfzLevels

    def getZSlice(self, slice_number):
        if slice_number >= self.getNumberOfSlices():
            raise Exception("The zSlice reference " + str(slice_number) + " is invalid as the set has only " + str(
                self.getNumberOfSlices()) + " zSlices.")
        else:
            return self.zSlices[slice_number]

    def setZSlices(self, zSlice, zLevel):
        self.zSlices[zLevel] = zSlice

    def getZValue(self, slice_number):
        if slice_number >= self.getNumberOfSlices():
            raise Exception("The zSlice reference " + str(slice_number) + " is invalid as the set has only " + str(
                self.getNumberOfSlices()) + " zSlices.")
        else:
            return self.slices_zValues[slice_number]

    def getName(self):
        return self.name

    def setZValues(self):
        stepSize = 1.0 / self.getNumberOfSlices()
        firstStep = stepSize
        self.slices_zValues = numpy.zeros(self.getNumberOfSlices()).tolist()
        for i in range(len(self.slices_zValues)):
            self.slices_zValues[i] = firstStep + i * stepSize

    def getFSWeightedAverage(self, x):
        numerator = 0.0
        denominator = 0.0
        for i in range(self.getNumberOfSlices()):
            numerator += self.getZSlice(i).getFSAverage(x) * self.getZValue(i)
            denominator += self.getZValue(i)

        return numerator / denominator

    def getFS(self, x: float):
        slice = T1MF_Discretized("VerticalSlice_at" + str(x) + "_of_" + self.getName())
        for i in range(self.numberOfzLevels):
            temp = self.getZSlice(i).getFS(x)
            slice.addPoint([self.getZValue(i), temp[0]])
            slice.addPoint([self.getZValue(i), temp[1]])
        if slice.getNumberOfPoints() > 0:
            return slice
        else:
            return Null

    def getZValues(self):
        if len(self.slices_zValues) == 0 or self.slices_zValues == Null:
            self.setZValues()
        return self.slices_zValues

    def getSupport(self):
        return self.support

    def isleftShoulder(self):
        return self.isLeftShoulder

    def isrightShoulder(self):
        return self.isLeftShoulder

    def setLeftShoulder(self, isLeftShoulder):
        self.isLeftShoulder = isLeftShoulder

    def setRightShoulder(self, isRightShoulder):
        self.isRightShoulder = isRightShoulder

    def getCentroid(self, primaryDiscretizationLevel: int):
        slice = T1MF_Discretized("Centroid_of_" + self.getName())
        for i in range(self.numberOfzLevels):
            temp = self.getZSlice(i).getCentroid(primaryDiscretizationLevel)
            slice.addPoint([self.getZValue(i), temp[0]])
            slice.addPoint([self.getZValue(i), temp[1]])
        if slice.getNumberOfPoints() > 0:
            return slice
        else:
            return Null

    def getPeak(self):
        average = 0
        for i in range(self.getNumberOfSlices()):
            average += self.getNumberOfSlices()
        average = average / self.getNumberOfSlices()

        return average


class T2MF_CylExtension(T2MF_Prototype):

    def __init__(self, baseSet: T1MF_Prototype, zDiscLevel: int):
        super().__init__("T2zCyl extension of " + baseSet.getName())
        self.__baseSet = baseSet
        self.__zDiscretizationlevel = zDiscLevel
        self.__zSpacing = 1.0 / self.__zDiscretizationlevel
        self.zSlices: [IT2MF_Cylinder] = []
        self.support = [float(-math.inf), float(math.inf)]
        self.slices_zValues = []

        for i in range(self.__zDiscretizationlevel):
            self.slices_zValues.append((i + 1) * self.__zSpacing)
            self.zSlices.append(
                IT2MF_Cylinder("Cyl-ext-at-" + str(self.slices_zValues[i]), baseSet.getAlphaCut(self.slices_zValues[i]),
                               None, None))

        self.numberOfzLevels = zDiscLevel

    def clone(self):
        return T2MF_CylExtension(self.__baseSet, self.__zDiscretizationlevel)


class T2MF_Gaussian(T2MF_Prototype):
    __primer: IT2MF_Gaussian

    def __init__(self, name: str, primer, numberOfzLevels):
        super().__init__(name)
        if type(primer) is list and numberOfzLevels is None:
            self.numberOfzLevels = len(primer)
            self.support = primer[0].getSupport()
            self.slices_fs = []
            self.slices_zValues = []
            self.z_stepSize = 1.0 / self.numberOfzLevels
            self.slices_zValues = []
            self.zSlices = primer
            for i in range(self.numberOfzLevels):
                self.slices_zValues.append(self.z_stepSize*(i+1))
        else:
            self.numberOfzLevels = numberOfzLevels
            self.__primer = primer
            self.support = [primer.getSupport()[0], primer.getSupport()[1]]
            self.slices_fs = []
            self.slices_zValues = []
            self.z_stepSize = 1.0 / numberOfzLevels
            self.zSlices = []
            stepsize_spread = (primer.getUMF().getSpread() - primer.getLMF().getSpread()) / (
                        (numberOfzLevels - 1) * 2.0)
            stepsize_mean = (primer.getUMF().getMean() - primer.getLMF().getMean()) / ((numberOfzLevels - 1) * 2.0)

            inner = [primer.getLMF().getMean(), primer.getLMF().getSpread()]
            outer = [primer.getUMF().getMean(), primer.getUMF().getSpread()]

            self.zSlices.append(IT2MF_Gaussian(primer.getName() + "_zSlice_0",
                                               T1MF_Gaussian(primer.getName() + "_zSlice_0_UMF", outer[0], outer[1]),
                                               T1MF_Gaussian(primer.getName() + "_zSlice_0_LMF", inner[0], inner[1])))
            if primer.isleftShoulder(): self.zSlices[0].setLeftShoulder(True)
            if primer.isrightShoulder(): self.zSlices[0].setRightShoulder(True)

            self.slices_zValues.append(self.z_stepSize)

            for i in range(1, numberOfzLevels):
                self.slices_zValues.append((i + 1) * self.z_stepSize)
                inner[0] += stepsize_mean
                inner[1] += stepsize_spread
                outer[0] += stepsize_mean
                outer[1] += stepsize_spread
                if outer[0] < inner[0]: inner[0] = outer[0]
                if outer[1] < inner[1]: inner[1] = outer[1]

                self.zSlices.append(IT2MF_Gaussian(primer.getName() + "_zSlice_" + str(i),
                                                   T1MF_Gaussian(primer.getName() + "_zSlice_" + str(i) + "_UMF",
                                                                 outer[0], outer[1]),
                                                   T1MF_Gaussian(primer.getName() + "_zSlice_" + str(i) + "_LMF",
                                                                 inner[0], inner[1])))
                if primer.isleftShoulder(): self.zSlices[i].setLeftShoulder(True)
                if primer.isrightShoulder(): self.zSlices[i].setRightShoulder(True)
                self.zSlices[i].setSupport(primer.getSupport())

    def clone(self):
        return T2MF_Gaussian(self.name, self.__primer, self.numberOfzLevels)

    def getZSlice(self, slice_number: int):
        return self.zSlices[slice_number]

    def setSupport(self, support):
        self.support = support
        for i in range(self.numberOfzLevels):
            self.zSlices[i].setSupport(self.getSupport())


class T2MF_Trapezoidal(T2MF_Prototype):
    __primer: IT2MF_Trapezoidal


    def __init__(self, name, primer, numberOfzLevels):
        super().__init__(name)
        if type(primer) is list and len(primer) == 2 and type(numberOfzLevels) is int:
            self.numberOfzLevels = numberOfzLevels
            self.support = primer[0].getSupport()
            self.slices_fs = []
            self.slices_zValues = []
            self.zSlices = [primer[0]]
            self.zSlices[0].setName(self.getName()+"_Slice_0")

            self.z_stepSize = 1.0/numberOfzLevels
            self.slices_zValues.append(self.z_stepSize)

            lsu = (primer[1].getUMF().getA() - primer[0].getUMF().getA())/(numberOfzLevels - 1.0)
            lsl = (primer[0].getLMF().getA() - primer[1].getLMF().getA()) / (numberOfzLevels - 1.0)

            rsu = (primer[0].getUMF().getC() - primer[1].getUMF().getC())/(numberOfzLevels - 1.0)
            rsl = (primer[1].getLMF().getC() - primer[0].getLMF().getC()) / (numberOfzLevels - 1.0)

            inner = [primer[0].getLMF().getA(), primer[0].getLMF().getB(), primer[0].getLMF().getC(), 0]
            outer = [primer[0].getUMF().getA(), primer[0].getUMF().getB(), primer[0].getUMF().getC(), 0]

            for i in range(1, numberOfzLevels-1):
                self.slices_zValues.append(self.slices_zValues[i-1]+self.z_stepSize)
                inner[0] -= lsl; inner[3] += rsl
                outer[0] += lsu; outer[3] -= rsu
                self.zSlices.append(IT2MF_Trapezoidal("Slice "+str(i), T1MF_Trapezoidal("upper_slice "+str(i),outer), T1MF_Trapezoidal("lower_slice "+str(i),inner)))

            self.slices_zValues.append(1.0)
            self.zSlices.append(primer[1])

        elif type(primer) is list and numberOfzLevels is None:
            self.numberOfzLevels = len(primer)
            self.support = primer[0].getSupport()
            self.slices_fs = []
            self.slices_zValues = []
            self.z_stepSize = 1.0 / self.numberOfzLevels
            self.zSlices = primer
            for i in range(self.numberOfzLevels):
                self.slices_zValues.append(self.z_stepSize*(i+1))
        else:
            stepsize = numpy.zeros(4).tolist()
            self.numberOfzLevels = numberOfzLevels
            self.support = primer.getSupport()
            self.__primer = primer
            self.slices_fs = []
            self.slices_zValues = []
            self.z_stepSize = 1.0/numberOfzLevels
            stepsize[0] = (primer.getLMF().getA() - primer.getUMF().getA()) / ((numberOfzLevels - 1) * 2.0)
            stepsize[1] = (primer.getLMF().getB() - primer.getUMF().getB()) / ((numberOfzLevels - 1) * 2.0)
            stepsize[2] = (primer.getUMF().getC() - primer.getLMF().getC()) / ((numberOfzLevels - 1) * 2.0)
            stepsize[3] = (primer.getUMF().getD() - primer.getLMF().getD()) / ((numberOfzLevels - 1) * 2.0)
            inner = primer.getLMF().getParameters()
            outer = primer.getUMF().getParameters()

            self.zSlices = []
            self.zSlices.append(IT2MF_Trapezoidal("Slice 0", primer.getUMF(), primer.getLMF()))

            self.slices_zValues.append(self.z_stepSize)
            for i in range(1, numberOfzLevels):
                self.slices_zValues.append((i+1)*self.z_stepSize)
                inner[0] -= stepsize[0]; inner[1] -= stepsize[1]; inner[2] += stepsize[2]; inner[3] += stepsize[3]
                outer[0] += stepsize[0]; outer[1] += stepsize[1]; outer[2] -= stepsize[2]; outer[3] -= stepsize[3]
                if inner[0] < outer[0]: inner[0] = outer[0]
                if inner[1] < outer[1]: inner[1] = outer[1]
                if inner[2] > outer[2]: inner[2] = outer[2]
                if inner[3] > outer[3]: inner[3] = outer[3]
                self.zSlices.append(IT2MF_Trapezoidal("Slice "+str(i), T1MF_Trapezoidal("upper_slice "+str(i),outer), T1MF_Trapezoidal("lower_slice "+str(i),inner)))

    def clone(self):
        print("Cloning for T2MF_Trapezoidal needs to be re-implemented.")
        return Null

    def getZSlice(self, slice_number):
        return self.zSlices[slice_number]

    def getLeftShoulderStart(self):
        print("Shoulder methods not implemented!")
        return float(math.nan)

    def getRightShoulderStart(self):
        print("Shoulder methods not implemented!")
        return float(math.nan)


class T2MF_Triangular(T2MF_Prototype):

    __primer:IT2MF_Triangular

    def __init__(self, name, primer, numberOfzLevels):
        super().__init__(name)
        if type(primer) is list and len(primer) == 2 and type(numberOfzLevels) is int:
            self.numberOfzLevels = numberOfzLevels
            self.support = primer[0].getSupport()
            self.z_stepSize = 1.0 / numberOfzLevels
            self.slices_fs = []
            self.slices_zValues = [self.z_stepSize]
            self.zSlices = [primer[0]]

            lsu = (primer[1].getUMF().getStart() - primer[0].getUMF().getStart()) / (numberOfzLevels - 1.0)
            lsl = (primer[0].getLMF().getStart() - primer[1].getLMF().getStart()) / (numberOfzLevels - 1.0)

            rsu = (primer[0].getUMF().getEnd() - primer[1].getUMF().getEnd()) / (numberOfzLevels - 1.0)
            rsl = (primer[1].getLMF().getEnd() - primer[0].getLMF().getEnd()) / (numberOfzLevels - 1.0)

            inner = [primer[0].getLMF().getStart(), primer[0].getLMF().getEnd(),primer[0]. getLMF().getEnd()]
            outer = [primer[0].getUMF().getStart(), primer[0].getUMF().getEnd(),primer[0]. getUMF().getEnd()]

            for i in range(1, numberOfzLevels-1):
                self.slices_zValues.append((i+1)*self.z_stepSize)
                inner[0] -= lsl; inner[2] += rsl
                outer[0] += lsu; outer[2] -= rsu

                self.zSlices.append(IT2MF_Triangular(self.getName()+"_zSlice_"+str(i), T1MF_Triangular(self.getName()+"_zSlice_"+str(i)+"_UMF", outer[0], outer[1],outer[2]), T1MF_Triangular(self.getName()+"_zSlice_"+str(i)+"_LMF", inner[0], inner[1],inner[2])))
                self.zSlices[i].setSupport(primer[0].getSupport())

            self.slices_zValues.append(1.0)
            self.zSlices.append(primer[1])

        elif type(primer) is list and numberOfzLevels is None:
            self.numberOfzLevels = len(primer)
            self.support = primer[0].getSupport()
            self.slices_fs = []
            self.slices_zValues = []
            self.z_stepSize = 1.0 / self.numberOfzLevels
            self.zSlices = primer
            for i in range(self.numberOfzLevels):
                self.slices_zValues.append(self.z_stepSize * (i + 1))
        else:
            self.numberOfzLevels = numberOfzLevels
            self.support = primer.getSupport()
            self.__primer = primer
            self.z_stepSize = 1.0 / numberOfzLevels
            self.slices_fs = []
            self.slices_zValues = [self.z_stepSize]
            self.zSlices = [IT2MF_Triangular("Slice 0", primer.getUMF(), primer.getLMF())]

            left_stepsize = (primer.getLMF().getStart() - primer.getUMF().getStart())/((numberOfzLevels-1)*2.0)
            right_stepsize = (primer.getUMF().getEnd() - primer.getLMF().getEnd())/((numberOfzLevels-1)*2.0)

            inner = [primer.getLMF().getStart(), primer.getLMF().getPeak(), primer.getLMF().getEnd()]
            outer = [primer.getUMF().getStart(), primer.getUMF().getPeak(), primer.getUMF().getEnd()]

            for i in range(1, numberOfzLevels):
                self.slices_zValues.append((i+1)*self.z_stepSize)
                inner[0] -= left_stepsize; inner[2] += right_stepsize
                outer[0] += left_stepsize; outer[2] -= right_stepsize
                if abs(inner[0] - outer[0]) < 0.00001:
                    outer[0] = inner[0]
                if abs(inner[2] - outer[2]) < 0.00001:
                    outer[2] = inner[2]

                self.zSlices.append(IT2MF_Triangular("Slice "+str(i), T1MF_Triangular("Slice_"+str(i)+"_UMF", outer[0], outer[1], outer[2]), T1MF_Triangular("Slice_"+str(i)+"_LMF", inner[0], inner[1], inner[2])))


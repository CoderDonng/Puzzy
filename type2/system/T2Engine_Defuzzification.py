import numpy
from datashape import Null

from intervaltype2.operation.IT2Engine_Centroid import IT2Engine_Centroid
from type1.operation.T1MF_Discretized import T1MF_Discretized
from type2.operation.T2MF_Discretized import T2MF_Discretized


class T2Engine_Defuzzification:
    __tRSet: T1MF_Discretized
    __dset: T2MF_Discretized
    __dPoints:[]
    __dPoints_real:[]
    __crisp_output:float

    __MINIMUM = 0
    __PRODUCT = 0
    __tnorm = __MINIMUM
    __DEBUG_S = False

    def __init__(self, primaryDiscretizationLevel: int):

        self.I2EC = IT2Engine_Centroid(primaryDiscretizationLevel)

    def typeReduce(self, set):

        if set == Null:
            return Null

        dividend_left = 0.0; dividend_right = 0.0
        divisor_left = 0.0; divisor_right = 0.0

        for i in range(set.getNumberOfSlices()):
            if set.getZSlice(i) == Null:
                print("slice "+str(i)+" is null.")
            else:
                centroid = self.I2EC.getCentroid(set.getZSlice(i))
                if centroid != Null:
                    dividend_left += centroid[0] * set.getZValue(i)
                    dividend_right += centroid[1] * set.getZValue(i)

                    divisor_left += set.getZValue(i)
                    divisor_right += set.getZValue(i)

        return [dividend_left/divisor_left, dividend_right/divisor_right]

    def typeReduce_standard(self, set, xResolution, yResolution):
        self.__dset = T2MF_Discretized(set, xResolution, yResolution)
        self.__dPoints_real = [None for i in range(xResolution)]
        temp = [None for i in range(yResolution)]
        for i in range(xResolution):
            counter = 0
            for j in range(yResolution):
                if self.__dset.getSetDataAt(i,j) > 0:
                    temp[counter] = [self.__dset.getSetDataAt(i,j), self.__dset.getDiscY(j)]
                    counter += 1

            List = []
            for k in range(counter):
                List.append(temp[k])
            self.__dPoints_real[i] = List

        if self.__DEBUG_S:
            print("Number of vertical slices: " +str(len(self.__dPoints_real)))
            print("Vertical Slice Positions on x-Axis: ")
            for i in range(xResolution):
                print("Slice " + str(i) + " is at x = " + str(self.__dset.getPrimaryDiscretizationValues()[i]))

            print("Actual Slices:")
            self.printSlices(self.__dPoints_real)

        number_of_rows = 0
        for i in range(len(self.__dPoints_real)):
            if len(self.__dPoints_real[i]) != 0:
                if number_of_rows == 0:
                    number_of_rows = len(self.__dPoints_real[i])
                else:
                    number_of_rows *= len(self.__dPoints_real[i])

        if self.__DEBUG_S:
            print("Final array contains "+ str(number_of_rows)+" rows!")

        wavySlices = [[Null for i in range(xResolution)] for j in range(number_of_rows)]

        for i in range(xResolution):
            counter = 0
            for k in range(number_of_rows):
                if len(self.__dPoints_real[i]) != 0:
                    wavySlices[k][i] = (self.__dPoints_real[i])[counter]
                else:
                    print("Setting wavy slice to null!")

                counter += 1
                if counter == len(self.__dPoints_real[i]):
                    counter = 0

        if self.__DEBUG_S:
            print("Wavy Slices:")
            self.printSlices(wavySlices)

        wavycentroids = numpy.zeros(number_of_rows).tolist()

        for i in range(number_of_rows):
            dividend = 0; divisor = 0
            for j in range(xResolution):
                if wavySlices[i][j] == Null:
                    if self.__DEBUG_S:
                        print("Skipping wavy slice " + str(i) + " as its not defined at " + str(j))
                else:
                    dividend += self.__dset.getPrimaryDiscretizationValues()[j] * wavySlices[i][j][1]
                    divisor += wavySlices[i][j][1]

            if self.__DEBUG_S:
                print("wavySlices - Dividend: "+str(dividend)+"  Divisior: "+str(divisor))
            wavycentroids[i] = dividend/divisor
            if self.__DEBUG_S:
                print("Centroid of wavyslice "+str(i)+" is: "+str(wavycentroids[i]))

        if self.__DEBUG_S:
            print("Final type-reduced tuples:")
        Min = 1.0
        reduced = [None for i in range(number_of_rows)]
        for i in range(number_of_rows):
            if self.__tnorm == self.__MINIMUM:
                Min =1.0
                for j in range(xResolution):
                    if wavySlices[i][j] is not Null:
                        Min = min(Min, wavySlices[i][j][0])
            elif self.__tnorm == self.__PRODUCT:
                Min = 1.0
                for j in range(xResolution):
                    if wavySlices[i][j] is not Null:
                        Min *= wavySlices[i][j][0]
            reduced[i] = [Min, wavycentroids[i]]
            if self.__DEBUG_S:
                print(str(reduced[i]))
            print(str(reduced[i][1])+","+str(reduced[i][0]))

        self.__tRSet = T1MF_Discretized("output")
        self.__tRSet.addPoints(reduced)

        dividend = 0; divisor = 0
        for i in range(len(reduced)):
            dividend += reduced[i][0] * reduced[i][1]
            divisor += reduced[i][0]
        self.__crisp_output = dividend/divisor
        return self.__crisp_output

    def printSlices(self, o:[]):
        for i in range(len(o)):
            print("Slice " + str(i) + " , with a length of: " + str(len(o[i])))
            for j in range(len(o[i])):
                if o[i][j] is not None or o[i][j] is not Null:
                    print("Point " + str(j) + ": " + o[i][j][0] + "/" + o[i][j][1] + " ")
                else:
                    print("Null")





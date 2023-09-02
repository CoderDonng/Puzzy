import random

from datashape import Null

from CIT2.operation.TupleOperation import TupleOperations
from intervaltype2.IT2MF import IT2MF_Prototype
from type1.T1MF import T1MF_Prototype


class CIT2(IT2MF_Prototype):

    def __init__(self, name, generator_set, dis):
        super().__init__(name, None,None)
        if type(dis) is list:
            if dis[0]>0 or dis[1]<0:
                raise Exception("The displacement interval must be of the form [a,b] with a<=0 and b>=0")
            self.__generatorSet = generator_set
            self.__displacementInterval = dis
            self.__leftmostAES = self.__generatorSet.shiftFunction(self.__generatorSet.getName()+" shifted by"+str(self.__displacementInterval[0]), self.__displacementInterval[0])
            self.__rightmostAES = self.__generatorSet.shiftFunction(
                self.__generatorSet.getName() + " shifted by" + str(self.__displacementInterval[1]),
                self.__displacementInterval[1])

            self.initializeUpperbound()
            self.initializeLowerbound()
            generator_support = generator_set.getSupport()
            self.support = [generator_support[0]+self.__displacementInterval[0], generator_support[1]+self.__displacementInterval[1]]
        else:
            displacementInterval = [dis*-1, dis]
            if displacementInterval[0]>0 or displacementInterval[1]<0:
                raise Exception("The displacement interval must be of the form [a,b] with a<=0 and b>=0")

            self.__generatorSet = generator_set
            self.__displacementInterval = displacementInterval
            self.__leftmostAES = self.__generatorSet.shiftFunction(self.__generatorSet.getName() + " shifted by" + str(self.__displacementInterval[0]), self.__displacementInterval[0])
            self.__rightmostAES = self.__generatorSet.shiftFunction(
                self.__generatorSet.getName() + " shifted by" + str(self.__displacementInterval[1]),
                self.__displacementInterval[1])

            self.initializeUpperbound()
            self.initializeLowerbound()
            generator_support = generator_set.getSupport()
            self.support = [generator_support[0] + self.__displacementInterval[0],
                            generator_support[1] + self.__displacementInterval[1]]

    def getDisplacementInterval(self): return self.__displacementInterval

    def setSupport(self, support):
        self.support = support
        self.__rightmostAES.setSupport(TupleOperations().intersection(support, self.__rightmostAES.getSupport()))
        self.__leftmostAES.setSupport(TupleOperations().intersection(support, self.__leftmostAES.getSupport()))
        if self.uMF is not None:
            self.uMF.setSupport(TupleOperations().intersection(support, self.uMF.getSupport()))
        if self.lMF is not None:
            self.lMF.setSupport(TupleOperations().intersection(support, self.lMF.getSupport()))

    def getGeneratorSet(self): return self.__generatorSet

    def getLeftmostAES(self): return self.__leftmostAES

    def getRightmostAES(self): return self.__rightmostAES

    def initializeUpperbound(self):
        self.uMF = CIT2_Boundary(self.name, "UPPERBOUND", self.__generatorSet, self.__leftmostAES, self.__rightmostAES, self.__displacementInterval)

    def initializeLowerbound(self):
        self.lMF = CIT2_Boundary(self.name, "LOWERBOUND", self.__generatorSet, self.__leftmostAES, self.__rightmostAES, self.__displacementInterval)

    def getCentroid(self, discretization: int):
        generator_centroid = self.__generatorSet.getDefuzzifiedCentroid(discretization)
        return [generator_centroid+self.__displacementInterval[0], generator_centroid+self.__displacementInterval[1]]

    def toIT2(self):
        return IT2MF_Prototype("IT2_from_"+self.name, self.uMF, self.lMF)

    def getRandomAES(self):
        if self.__displacementInterval[0] == self.__displacementInterval[1]:
            return self.__generatorSet
        randomValue = random.uniform(self.__displacementInterval[0], self.__displacementInterval[1])
        return self.__generatorSet.shiftFunction("", randomValue)


class CIT2_Boundary(T1MF_Prototype):

    def __init__(self, name, type, generator_set, leftmost_aes, rightmost_aes, displacement_interval):
        super().__init__(name)
        self.__type = type
        self.__leftmostAES = leftmost_aes
        self.__rightmostAES = rightmost_aes
        self.__generatorSet = generator_set
        self.__displacementInterval = displacement_interval
        self.support = [leftmost_aes.getSupport()[0], rightmost_aes.getSupport()[1]]

    def getFS(self, x):
        if self.__type == "LOWERBOUND":
            interval_to_check = self.__generatorSet.getMinPoints()
        else:
            interval_to_check = self.__generatorSet.getMaxPoints()
        min_max_point = self.inIntervals(x, interval_to_check, self.__type)
        if self.__type == "LOWERBOUND":
            partial_result = min(self.__leftmostAES.getFS(x), self.__rightmostAES.getFS(x))
            if min_max_point != -1:
                return min(partial_result, min_max_point)
            return partial_result
        else:
            partial_result = max(self.__leftmostAES.getFS(x), self.__rightmostAES.getFS(x))
            if min_max_point != -1:
                return max(partial_result, min_max_point)
            return partial_result

    def inIntervals(self,x,intervals,boundary_type):
        if intervals == Null:
            return -1
        final_membership_value = -1
        for cur in iter(intervals): #若intervals 为 [[start, start], [end, end]]
            if cur[0]+self.__displacementInterval[0] <= x <= cur[1]+self.__displacementInterval[1]: #第一次循环 start-dis<x<start+dis
                current_membership_degree = self.__generatorSet.getFS(cur[0])
                if boundary_type == "LOWERBOUND" and (current_membership_degree < final_membership_value or final_membership_value == -1):
                    final_membership_value = current_membership_degree
                if boundary_type == "UPPERBOUND" and (current_membership_degree > final_membership_value or final_membership_value == -1):
                    final_membership_value = current_membership_degree

        return final_membership_value

    def getAlphaCut(self, param):
        raise Exception("Not supported yet.")

    def getPeak(self):
        raise Exception("Not supported yet.")

    def compareTo(self, o:object):
        raise Exception("Not supported yet.")

    def clone(self):
        return CIT2_Boundary(self.name, self.__type, self.__generatorSet, self.__leftmostAES, self.__rightmostAES, self.__displacementInterval)



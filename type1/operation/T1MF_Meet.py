from datashape import Null
from type1.T1MF import T1MF_Prototype


class T1MF_Meet(T1MF_Prototype):
    __intersectionExists = False
    __v1: float; __v2: float; temp: float; temp2: float
    __resolution = 30; __alphaCutDiscLevel = 10000; __max_resolution = 10000

    def __init__(self, a, b):
        super().__init__("T1MF_Meet")
        if a == Null or b == Null:
            self.__intersectionExists = False
        else:
            self.name = a.getName()+"<meet>"+b.getName()
            self.__intersectionExists = True
            self.temp = self.findMax(a)
            self.temp2 = self.findMax(b)

            self.support = [min(a.getSupport()[0],b.getSupport()[0]),min(a.getSupport()[1],b.getSupport()[1])]

            if self.temp < self.temp2:
                self.__v1 = self.temp
                self.__v2 = self.temp2
                self.__f1 = a
                self.__f2 = b
            else:
                self.__v1 = self.temp2
                self.__v2 = self.temp
                self.__f1 = b
                self.__f2 = a

    def findMax(self, set):
        stepSize = (set.getSupport()[1]-set.getSupport()[0])/(self.__max_resolution-1)
        currentMax = 0.0; temp =0.0; maxStep = 0.0
        currentStep = set.getSupport()[0]
        for i in range(self.__max_resolution):
            temp = set.getFS(currentStep)
            if temp == 1:
                return currentStep
            if temp >= currentMax:
                currentMax = temp
                maxStep = currentStep
            currentStep += stepSize

        return maxStep

    def getFS(self, x: float):
        if x < self.__v1:
            return max(self.__f1.getFS(x), self.__f2.getFS(x))
        else:
            if x < self.__v2:
                return self.__f1.getFS(x)
            else:
                return min(self.__f1.getFS(x), self.__f2.getFS(x))

    def intersectionExists(self):
        return self.__intersectionExists

    def getAlphaCut(self, alpha: float):
        stepSize:float = (self.getSupport()[1]-self.getSupport()[0])/(self.__alphaCutDiscLevel-1.0)
        left = 0.0; right = 0.0

        currentStep: float = self.getSupport()[0]
        n = 0
        while n < self.__alphaCutDiscLevel:
            temp = abs(self.getFS(currentStep) - alpha)
            if temp < 0.001:
                left = currentStep
                break
            currentStep += stepSize
            n+=1

        currentStep = self.getSupport()[1]
        n=0
        while n < self.__alphaCutDiscLevel:
            temp = abs(self.getFS(currentStep) - alpha)
            if temp < 0.001:
                right = currentStep
                break
            currentStep -= stepSize
            n+=1

        alphaCut = [left, right]
        return alphaCut

    def getPeak(self):
        raise Exception("Not supported yet.")

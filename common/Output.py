import numpy

class Output:
    __name:str
    __domain:[float, float]
    __discretisationLevel = 100
    __discretisedDomain = []

    def __init__(self, name:str, domain:[float, float]):
        self.__name = name
        self.__domain = domain

    def getName(self): return self.__name

    def setName(self, name:str): self.__name = name

    def getDiscretisationLevel(self): return self.__discretisationLevel

    def setDiscretisationLevel(self,discretisationLevel:int): self.__discretisationLevel = discretisationLevel

    def getDomain(self): return self.__domain

    def setDomain(self, domain: [float, float]): self.__domain = domain

    def getDiscretizations(self):

        if len(self.__discretisedDomain) == 0 or len(self.__discretisedDomain) != self.__discretisationLevel:
            l = numpy.zeros(self.__discretisationLevel).tolist()
            self.__discretisedDomain = l
            stepSize = (self.__domain[1]-self.__domain[0])/(self.__discretisationLevel-1.0)
            self.__discretisedDomain[0] = self.__domain[0]
            self.__discretisedDomain[self.__discretisationLevel-1] = self.__domain[1]
            for i in range(1,self.__discretisationLevel-1):
                self.__discretisedDomain[i] = self.__domain[0]+i*stepSize

            return self.__discretisedDomain
        else:
            return self.__discretisedDomain

    def compareTo(self,o):
        if self.__name < o.getName():
            return -1
        elif self.__name == o.getName():
            return 0
        else:
            return 1




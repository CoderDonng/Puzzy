

class IT2_COSInferenceData:
    __firingStrength:[float, float]
    __centroidValue: float

    def __init__(self, f:list, c:float):
        self.__firingStrength = f
        self.__centroidValue = c

    def getFStrength(self): return self.__firingStrength

    def getSelectedCentroidEndpoint(self): return self.__centroidValue
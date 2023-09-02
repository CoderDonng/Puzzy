from type1.T1MF import T1MF_Prototype


class T1MF_Intersection(T1MF_Prototype):

    def __init__(self,name, setA, setB):
        super().__init__(name)
        self.support = [max(setA.getSupport()[0], setB.getSupport()[0]), min(setA.getSupport()[1], setB.getSupport()[1])]
        self.__setA = setA
        self.__setB = setB

    def getFS(self, x: float):
        return min(self.__setA.getFS(x), self.__setB.getFS(x))

    def getAlphaCut(self, alpha: float):
        raise Exception("Not supported yet.")

    def getPeak(self):
        raise Exception("Not supported yet.")


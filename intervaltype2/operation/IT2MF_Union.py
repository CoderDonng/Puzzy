from intervaltype2.IT2MF import IT2MF_Prototype
from type1.operation.T1MF_Union import T1MF_Union


class IT2MF_Union(IT2MF_Prototype):

    __sets: list
    __isNull = False

    def __init__(self, a, b):
        super().__init__("Union of ("+a.getName()+" and "+b.getName()+")", None, None)
        self.uMF = T1MF_Union(a.getUMF(),b.getUMF())
        self.lMF = T1MF_Union(a.getLMF(),b.getLMF())
        self.support = [min(a.getSupport()[0],b.getSupport()[0]),max(a.getSupport()[1],b.getSupport()[1])]

    def getSets(self): return self.__sets

    def isNull(self): return self.__isNull

    def getPeak(self):
        raise Exception("Not supported yet.")
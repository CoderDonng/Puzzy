from intervaltype2.IT2MF import IT2MF_Prototype, IT2MF_Cylinder
from type1.operation.T1MF_Intersection import T1MF_Intersection


class IT2MF_Intersection(IT2MF_Prototype):
    __sets:[]
    __intersectionExists = False

    def __init__(self,a,b):
        super().__init__("dummy-intersection",None,None)
        if isinstance(a,IT2MF_Cylinder) or isinstance(b,IT2MF_Cylinder):
            if isinstance(a,IT2MF_Cylinder) and a.getUpperBound(0)==0.0:
                self.__intersectionExists = False
            elif isinstance(b,IT2MF_Cylinder) and b.getUpperBound(0)==0.0:
                self.__intersectionExists = False
            else:
                self.__intersectionExists = True
        elif a.getSupport()[0]==b.getSupport()[0]:
            self.__intersectionExists = True
        elif a.getSupport()[0]<b.getSupport()[0]:
            if a.getSupport()[1]>=b.getSupport()[0]:
                self.__intersectionExists = True
        elif a.getSupport()[0]<=b.getSupport()[1]:
            self.__intersectionExists = True

        if self.__intersectionExists:
            self.__sets = []
            if isinstance(a,IT2MF_Intersection):
                self.__sets.extend(a.getSets())
            else:
                self.__sets.append(a)

            if isinstance(b, IT2MF_Intersection):
                self.__sets.extend(b.getSets())
            else:
                self.__sets.append(b)

            self.uMF = T1MF_Intersection("uMF of Intersection of ("+a.getName()+","+b.getName()+")", a.getUMF(),b.getUMF())
            self.lMF = T1MF_Intersection("lMF of Intersection of ("+a.getName()+","+b.getName()+")", a.getLMF(),b.getLMF())
            self.__sets.reverse()
            set = self.__sets[0]
            if set.getSupport() != None and not isinstance(set,IT2MF_Cylinder):
                self.support = [set.getSupport()[0], set.getSupport()[1]]
                self.name = "Intersection of ("+set.getName()

            for i in range(1,len(self.__sets)):
                set = self.__sets[i]
                if not isinstance(set,IT2MF_Cylinder):
                    if self.support == None:
                        self.support = [set.getSupport()[0], set.getSupport()[1]]
                        self.name = "Intersection of ("+set.getName()
                    else:
                        self.support[0] = min(self.support[0], set.getSupport()[0])
                        self.support[1] = min(self.support[1], set.getSupport()[1])
                self.name += " and "+set.getName()
            self.name += ")"
        else:
            self.support = None

    def getSets(self): return self.__sets

    def containsSet(self, set): return set in self.__sets

    def getFS(self, x):
        if not self.__intersectionExists:
            return None
        else:
            returnValue = [1,1]
            setFS = None
            for set in iter(self.__sets):
                setFS = set.getFS(x)
                returnValue[0] = min(returnValue[0], setFS[0])
                returnValue[1] = min(returnValue[1], setFS[1])

            return returnValue

    def intersectionExists(self): return self.__intersectionExists





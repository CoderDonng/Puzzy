from datashape import Null

from intervaltype2.operation.IT2MF_Intersection import IT2MF_Intersection
from type2.operation.T2MF_Intersection import T2MF_Intersection


class T2Engine_Intersection:

    __TRADITIONAL = 0
    __GEOMETRIC = 1
    __intersection_method = __TRADITIONAL
    __intersection: T2MF_Intersection

    def __init__(self):
        pass

    def getIntersection(self, set_a, set_b):
        if set_a is Null or set_b is Null:
            return Null

        if set_a.getNumberOfSlices() != set_b.getNumberOfSlices():
            raise Exception("Both sets need to have the same number of slices to calculate their intersection!\n" +
                    "Here, set A ("+set_a.getName()+") has "+str(set_a.getNumberOfSlices())+" slices and set B ("+set_b.getName()+") has : "+str(set_b.getNumberOfSlices()))

        if self.__intersection_method == self.__TRADITIONAL:
            zSlices = [Null for i in range(set_a.getNumberOfSlices())]
            for i in range(set_a.getNumberOfSlices()):
                zSlices[i] = IT2MF_Intersection(set_a.getZSlice(i), set_b.getZSlice(i))
                if not zSlices[i].intersectionExists():
                    zSlices[i] = Null
            self.__intersection = T2MF_Intersection("Intersection of "+set_a.getName()+" and "+set_b.getName(), set_a.getNumberOfSlices(), set_a.getZValues(), zSlices)

        return self.__intersection

from datashape import Null

from intervaltype2.operation.IT2MF_Union import IT2MF_Union
from type2.operation.T2MF_Union import T2MF_Union


class T2Engine_Union:
    __TRADITIONAL = 0
    __GEOMETRIC = 1
    __union_method = __TRADITIONAL
    __union: T2MF_Union

    def __init__(self):
        pass

    def getUnion(self, set_a, set_b):
        if set_a is Null:
            return set_b
        if set_b is Null:
            return set_a
        if set_a.getNumberOfSlices() != set_b.getNumberOfSlices():
            raise Exception("Both sets need to have the same number of slices to calculate their intersection!")

        if self.__union_method == self.__TRADITIONAL:
            if set_a is Null and set_b is Null:
                return Null
            else:
                if set_a is Null:
                    return set_b
                else:
                    if set_b is Null:
                        return set_a
                    else:
                        zSlices = [Null for i in range(set_a.getNumberOfSlices())]
                        for i in range(set_a.getNumberOfSlices()):
                            zSlices[i] = IT2MF_Union(set_a.getZSlice(i),set_b.getZSlice(i))
                        self.__union = T2MF_Union("Union of "+set_a.getName()+" and "+set_b.getName(), set_a.getNumberOfSlices(), set_a.getZValues(),zSlices)
        elif self.__union_method == self.__GEOMETRIC:
            print("GEOMETRIC UNION IS NOT YET IMPLEMENTED!!!")

        return self.__union

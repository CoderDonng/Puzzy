import math

from datashape import Null

from type1.operation.T1MF_Discretized import T1MF_Discretized
from type2.T2MF import T2MF_Prototype


class T2MF_Union(T2MF_Prototype):

    def __init__(self, name, numberOfzLevels, slices_zValues, zSlices):
        super().__init__(name)
        self.numberOfzLevels = numberOfzLevels
        self.slices_zValues = slices_zValues
        self.zSlices = zSlices
        self.support = zSlices[0].getSupport()

    def clone(self):
        return T2MF_Union(self.name, self.numberOfzLevels, self.slices_zValues, self.zSlices)

    def getFS(self, x: float):
        slice = T1MF_Discretized("VSlice")
        for i in range(len(self.zSlices)):
            temp = self.getZSlice(i).getFS(x)
            if temp[1] == 0:
                if i == 0: slice = Null
                break
            else:
                slice.addPoint([self.getZValue(i), temp[0]])
                slice.addPoint([self.getZValue(i), temp[1]])

        return slice

    def isleftShoulder(self):
        print("Shoulder methods not implemented!")
        return False

    def isrightShoulder(self):
        print("Shoulder methods not implemented!")
        return False

    def getLeftShoulderStart(self):
        print("Shoulder methods not implemented!")
        return float(math.nan)

    def getRightShoulderStart(self):
        print("Shoulder methods not implemented!")
        return float(math.nan)

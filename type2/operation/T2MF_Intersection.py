from type2.T2MF import T2MF_Prototype


class T2MF_Intersection(T2MF_Prototype):

    def __init__(self, name, numberOfZlevels, slices_zValues, zSlices):
        super().__init__(name)
        self.numberOfzLevels = numberOfZlevels
        self.slices_zValues = slices_zValues
        self.zSlices = zSlices
        self.support = zSlices[0].getSupport()

    def clone(self):
        return T2MF_Intersection(self.name, self.numberOfzLevels, self.slices_zValues, self.zSlices)
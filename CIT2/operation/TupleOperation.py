import math
from datashape import Null


class TupleOperations:

    def __init__(self):
        pass

    def intersection(self, a: [float, float], b: [float, float]):
        if math.isnan(a[1]) and math.isnan(b[1]):
            return Null
        else:
            if math.isnan(a[1]):
                return b
            if math.isnan(b[1]):
                return a
            if max(a[0], b[0]) <= min(a[1], b[1]):
                return [max(a[0], b[0]), min(a[1], b[1])]
            else:
                return Null

    def union(self, a, b):
        if math.isnan(a[1]):
            return b
        if math.isnan(b[1]):
            return a
        return [min(a[0], b[0]), max(a[1], b[1])]
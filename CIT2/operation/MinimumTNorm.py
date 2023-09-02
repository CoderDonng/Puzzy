from datashape import Null


class MinimumTNorm:
    __instance = Null

    def __init__(self):
        pass

    def doTnorm(self, values:[[float,float]]):
        result:[float,float] = [1.0, 1.0]
        for it in iter(values):
            result[0] = min(result[0], it[0])
            result[1] = min(result[1], it[1])

        return result

    def getInstance(self):

        if self.__instance == Null:
            self.__instance = MinimumTNorm()
        return self.__instance

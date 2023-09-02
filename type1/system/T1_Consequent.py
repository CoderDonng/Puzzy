from common.Output import Output


class T1_Consequent:
    __name: str
    __output: Output

    def __init__(self, name: str, mF, output: Output):
        self.__name = name
        self.__mF = mF
        self.__output = output

    def setMF(self, mF): self.__mF = mF

    def getMF(self): return self.__mF

    def getOutput(self): return self.__output

    def setOutput(self, output: Output): self.__output = output

    def getName(self): return self.__name

    def setName(self, name): self.__name = name

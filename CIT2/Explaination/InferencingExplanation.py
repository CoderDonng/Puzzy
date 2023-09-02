class InferencingExplanation:

    __MFName: str
    __variableName: str
    __inferencingValue: [float, float]

    def __init__(self, mf_name: str, variable_name: str, inferencing_value: [float, float]):

        self.__MFName = mf_name
        self.__variableName = variable_name
        self.__inferencingValue = inferencing_value

    def getMFName(self): return self.__MFName

    def getVariableName(self): return self.__variableName

    def getInferencingValue(self): return self.__inferencingValue
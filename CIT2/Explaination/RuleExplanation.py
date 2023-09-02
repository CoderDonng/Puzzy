from CIT2.Explaination.InferencingExplanation import InferencingExplanation
from CIT2.system.CIT2_Rule import CIT2_Rule
from common.Output import Output


class RuleExplanation:
    __rule: CIT2_Rule
    __inputMembershipDegree: [InferencingExplanation]
    __output: Output
    __isUsingUpperFiring: bool
    __ruleFiringUsed: float
    __ruleFiring: [float, float]
    __formattedRuleFiring: [float, float]
    firingLevelUsed: str

    def __init__(self, rule: CIT2_Rule, is_using_upper_firing: bool):

        self.__inputMembershipDegrees = []
        self.__isUsingUpperFiring = is_using_upper_firing
        self.__rule = rule

        for it in iter(rule.getAntecedents()):
            current_MF = it.getCIT2()
            current_membership_degree = current_MF.getFS(it.getInput().getInput())
            self.__inputMembershipDegrees.append(InferencingExplanation(current_MF.getName(), it.getInput().getName(), current_membership_degree))

        self.__output = rule.getConsequents()[0].getOutput()

        self.__ruleFiring = rule.getFiringStrength("MIN")

        if self.__isUsingUpperFiring:
            self.__ruleFiringUsed = self.__ruleFiring[1]
            self.firingLevelUsed = "UPPER"
        else:
            self.__ruleFiringUsed = self.__ruleFiring[0]
            self.firingLevelUsed = "LOWER"

        self.__formattedRuleFiring = [round(self.__ruleFiring[0], 2), round(self.__ruleFiring[1], 2)]


    def toStr(self):
        result = " "
        first = True
        for it in iter(self.__inputMembershipDegrees):
            if not first:
                result += " and "
            else:
                result += "obtained because "
                first = False

            result += it.getVariableName()+" is "+it.getMFName()+" ["+str(round(it.getInferencingValue()[0], 2))+", "+str(round(it.getInferencingValue()[1], 2))+"]"

        return result

    def getRule(self): return self.__rule

    def getOutput(self): return self.__output

    def getRuleFiringUsed(self): return self.__ruleFiringUsed

    def printFiringLevelUsed(self): return self.firingLevelUsed



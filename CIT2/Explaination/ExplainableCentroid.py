from datashape import Null
from CIT2.CIT2 import CIT2
from CIT2.Explaination.RuleExplanation import RuleExplanation
from type1.T1MF import T1MF_IFRS


class ExplainableCentroid:

    __constrainedCentroid: [float, float]
    __leftEndpointExplaination: [[CIT2, RuleExplanation]]
    __rightEndpointExplaination: [[CIT2, RuleExplanation]]

    def __init__(self, centroid_interval, fired_fou, left_endpoint_explanation, right_endpoint_explanation, min_es, max_es, output):
        if fired_fou is Null and left_endpoint_explanation is Null and right_endpoint_explanation is Null and min_es is Null and max_es is Null and output is Null:
            self.__constrainedCentroid = centroid_interval
        else:
            self.__constrainedCentroid = centroid_interval
            self.__leftEndpointExplaination = left_endpoint_explanation
            self.__rightEndpointExplaination = right_endpoint_explanation
            self.min_es = min_es
            self.max_es = max_es
            self.__output = output
            self.__firedFOU = fired_fou

    def addRuleExplaination(self, current_rule: [CIT2, RuleExplanation], text_explanation: str):
        if current_rule[1].getRuleFiringUsed() > 0:
            text_explanation += current_rule[0].getName()+": "+str(current_rule[1].getRuleFiringUsed())+" "+str(current_rule[1])+" using the "+str(current_rule[1].printFiringLevelUsed())+" membership degree of each input terms\n"

    def getMin_es(self): return self.min_es

    def getMax_es(self): return self.max_es

    def getConstrainedCentroid(self): return self.__constrainedCentroid

    def getLeftEndpointExplaination(self): return self.__leftEndpointExplaination

    def getRightEndpointExplaination(self): return self.__rightEndpointExplaination

    def getOutputPartitioning(self):
        rule_consequents = []
        for it in iter(self.__leftEndpointExplaination):
            rule_consequents.append(it[0])
        for it in iter(self.__rightEndpointExplaination):
            rule_consequents.append(it[0])

        return rule_consequents

    def getFiredAES(self, output_partitioning: [CIT2], isLeftExplanation: bool):

        if isLeftExplanation:
            explanations = self.__leftEndpointExplaination
        else:
            explanations = self.__rightEndpointExplaination

        fired_aes = []

        for it in iter(output_partitioning):
            current_explanation = self.findExplanation(it, explanations)
            current_consequent = current_explanation[0]

            if isLeftExplanation:
                aes = current_consequent.getLeftmostAES()
            else:
                aes = current_consequent.getRightmostAES()

            fired_aes.append(T1MF_IFRS("Inferenced "+current_consequent.getName(), aes, current_explanation[1].getRuleFiringUsed()))

        return fired_aes

    def findExplanation(self, set: CIT2, explanations: [CIT2, RuleExplanation]):

        for it in iter(explanations):
            if set == it[0]:
                return it

        return Null
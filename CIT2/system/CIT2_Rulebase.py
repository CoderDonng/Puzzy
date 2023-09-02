import math
from datashape import Null
from CIT2.CIT2 import CIT2
from CIT2.Explaination.ExplainableCentroid import ExplainableCentroid
from CIT2.Explaination.RuleExplanation import RuleExplanation
from intervaltype2.operation.IT2MF_Union import IT2MF_Union
from type1.T1MF import T1MF_IFRS
from type1.operation.T1MF_Union import T1MF_Union


class CIT2_Rulebase:
    __CIT2Rules = []

    def __init__(self, rules):
        if rules is not Null:
            for rule in iter(rules):
                self.__CIT2Rules.append(rule)

    def addRule(self, current_rule): self.__CIT2Rules.append(current_rule)

    def getRules(self): return self.__CIT2Rules

    def getExplainableFiringIntervals(self):
        firing_intervals = {}
        new_rules = [None, None]
        for current_rule in iter(self.__CIT2Rules):
            current_consequent = current_rule.getConsequents()[0].getCIT2()
            current_cuts = current_rule.getFiringStrength("MIN")
            if firing_intervals.get(current_consequent) is None:
                firing_intervals[current_consequent] = [current_cuts, [current_rule, current_rule]]
            else:
                old_map_value = firing_intervals.get(current_consequent)
                old_rules_generating_firing = old_map_value[1]
                old_firing_interval = old_map_value[0]
                new_rules[0] = old_rules_generating_firing[0]
                new_rules[1] = old_rules_generating_firing[1]
                if old_firing_interval[0] < current_cuts[0]:
                    old_firing_interval[0] = current_cuts[0]
                    new_rules[0] = current_rule
                if old_firing_interval[1] < current_cuts[1]:
                    old_firing_interval[1] = current_cuts[1]
                    new_rules[1] = current_rule
                firing_intervals[current_consequent] = [old_firing_interval, [new_rules[0], new_rules[1]]]
        return firing_intervals

    def inferenceOnConsequentAES(self, isLeftEndpoint: bool, curr_switch_index: int, current_index: int, set: CIT2, truncations:[float, float]):
        truncation_height = 0
        if isLeftEndpoint:
            chosen_es = set.getLeftmostAES()

            if current_index >= curr_switch_index:
                truncation_height = truncations[0]
            else:
                truncation_height = truncations[1]
        else:
            chosen_es = set.getRightmostAES()
            if current_index >= curr_switch_index:
                truncation_height = truncations[1]
            else:
                truncation_height = truncations[0]
        if truncation_height == 0:
            return Null
        return T1MF_IFRS(set.getName(), chosen_es, truncation_height)

    def doSwitchIndexDefuzzification(self, discretization: int, buildExplanation: bool):
        truncation_heights = self.getExplainableFiringIntervals()
        consequent_mfs = []
        for it in iter(truncation_heights.keys()):
            consequent_mfs.append(it)

        output = self.__CIT2Rules[0].getConsequents()[0].getOutput()
        consequent_mfs.sort(key=lambda x: x.getSupport()[0])
        min_aes = Null
        max_aes = Null
        left_endpoint_centroid = float(math.inf)
        right_endpoint_centroid = float(-math.inf)
        result = [float(math.nan), float(math.nan)]
        left_switch_index = -1
        right_switch_index = -1

        computingLeftEndpoint = False

        for i in range(2):
            for curr_switch_index in range(len(consequent_mfs)):
                current_es = Null
                for k in range(len(consequent_mfs)):
                    current_ct2 = consequent_mfs[k]
                    aes_after_inference = self.inferenceOnConsequentAES(computingLeftEndpoint, curr_switch_index, k, current_ct2, truncation_heights.get(current_ct2)[0])

                    if current_es == Null:
                        current_es = aes_after_inference
                    else:
                        if aes_after_inference != Null:
                            current_es = T1MF_Union(current_es, aes_after_inference)

                if current_es == Null:
                    continue

                current_centroid = current_es.getDefuzzifiedCentroid(discretization)

                if current_centroid <= left_endpoint_centroid:
                    left_endpoint_centroid = current_centroid
                    min_aes = current_es
                    left_switch_index = curr_switch_index

                if current_centroid >= right_endpoint_centroid:
                    right_endpoint_centroid = current_centroid
                    max_aes = current_es
                    right_switch_index = curr_switch_index

            computingLeftEndpoint = True

        if left_endpoint_centroid != float(math.inf):
            result[0] = left_endpoint_centroid
        else:
            result[0] = float(math.nan)

        if right_endpoint_centroid != float(-math.inf):
            result[1] = right_endpoint_centroid
        else:
            result[1] = float(math.nan)

        if max_aes != Null:
            max_aes.setName("Max AES")
        if min_aes != Null:
            min_aes.setName("Min AES")

        if buildExplanation:
            return self.buildExplanation(result, consequent_mfs, truncation_heights, left_switch_index, right_switch_index, min_aes, max_aes, output)
        else:
            return ExplainableCentroid(result, Null, Null, Null, Null,Null, Null)

    def buildExplanation(self, centroid_value:[float,float], consequent_mfs:[CIT2], firing_intervals, left_switch_index: int, right_switch_index: int, min_aes, max_aes, output):

        explaination_left_endpoint = []
        explaination_right_endpoint = []

        for i in range(left_switch_index):
            current_consequent = consequent_mfs[i]
            current_rule = firing_intervals.get(current_consequent)[1][1]
            explaination_left_endpoint.append([current_consequent, RuleExplanation(current_rule, True)])

        for i in range(left_switch_index, len(consequent_mfs)):
            current_consequent = consequent_mfs[i]
            current_rule = firing_intervals.get(current_consequent)[1][0]
            explaination_left_endpoint.append([current_consequent, RuleExplanation(current_rule, False)])

        for i in range(right_switch_index):
            current_consequent = consequent_mfs[i]
            current_rule = firing_intervals.get(current_consequent)[1][0]
            explaination_right_endpoint.append([current_consequent, RuleExplanation(current_rule, False)])

        for i in range(right_switch_index, len(consequent_mfs)):
            current_consequent = consequent_mfs[i]
            current_rule = firing_intervals.get(current_consequent)[1][1]
            explaination_right_endpoint.append([current_consequent, RuleExplanation(current_rule, True)])

        return ExplainableCentroid(centroid_value, self.getFiredFOU(), explaination_left_endpoint, explaination_right_endpoint, min_aes, max_aes, output)

    def getFiredFOU(self):
        total_fired_fou = Null
        for current_rule in iter(self.__CIT2Rules):
            current_rule_fired_fou = current_rule.getFiredFOU()
            if current_rule_fired_fou != Null:
                if total_fired_fou == Null:
                    total_fired_fou = current_rule_fired_fou
                else:
                    total_fired_fou = IT2MF_Union(total_fired_fou, current_rule_fired_fou)

        return total_fired_fou

    def explainableDefuzzification(self, discretization: int):

        return self.doSwitchIndexDefuzzification(discretization, True)

    def clear(self):
        self.__CIT2Rules.clear()

    def defuzzification(self):
        val = 0
        fssum = 0
        for current_rule in iter(self.__CIT2Rules):
            current_cuts = current_rule.getFiringStrength("MIN")
            fs = (current_cuts[0]+current_cuts[1])/2
            val = val + fs * current_rule.getConCentroid()
            fssum = fssum + fs
        if fssum == 0:
            return 0
        else:
            return val/fssum
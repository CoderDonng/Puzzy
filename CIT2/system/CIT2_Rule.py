from CIT2.system.CIT2_Antecedent import CIT2_Antecedent
from CIT2.system.CIT2_Consequent import CIT2_Consequent
from intervaltype2.IT2MF import Gen_IT2MF
from type1.T1MF import T1MF_IFRS


class CIT2_Rule:

    __AND_OPERATOR = "MIN"
    __antecedents: [CIT2_Antecedent]
    __consequents: [CIT2_Consequent]

    def __init__(self, antecednets: [CIT2_Antecedent], consequents):
        self.__antecedents = antecednets
        if type(consequents) is list:
            self.__consequents = consequents
        else:
            self.__consequents = [consequents]

    def getAntecedents(self): return self.__antecedents

    def getConsequents(self): return self.__consequents

    def getFiringStrength(self, t_norm):

        fuzzified_inputs = []
        for it in iter(self.__antecedents):
            current_fuzzified_value = [0, 0]
            current_fuzzified_value[1] = it.getCIT2().getUMF().getFS(it.getInput().getInput())
            current_fuzzified_value[0] = it.getCIT2().getLMF().getFS(it.getInput().getInput())
            fuzzified_inputs.append(current_fuzzified_value)

        if t_norm == "MIN":
            result = [1, 1]
            for it in iter(fuzzified_inputs):
                result[0] = min(result[0], it[0])
                result[1] = min(result[1], it[1])

            return result

    def getFiredFOU(self):
        current_consequent = self.__consequents[0].getCIT2()
        firing_interval = self.computeFiringInterval()
        fired_lowerbound = T1MF_IFRS("Fired "+current_consequent.getName(), current_consequent.getLMF(), firing_interval[0])
        fired_upperbound = T1MF_IFRS("Fired "+current_consequent.getName(), current_consequent.getUMF(), firing_interval[1])

        return Gen_IT2MF("Fired FOU", fired_upperbound, fired_lowerbound)


    def computeFiringInterval(self):
        result = [1, 1]
        for it in iter(self.__antecedents):
            current_upperbound_fs = it.getCIT2().getUMF().getFS(it.getInput().getInput())
            current_lowerbound_fs = it.getCIT2().getLMF().getFS(it.getInput().getInput())
            if self.__AND_OPERATOR == "MIN":
                result[1] = min(result[1], current_upperbound_fs)
                result[0] = min(result[0], current_lowerbound_fs)
            else:
                result[1] = result[1] * current_upperbound_fs
                result[0] = result[0] * current_lowerbound_fs

        return result

    def getConCentroid(self):
        return self.__consequents[0].getCentroid()
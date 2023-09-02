import time
from intervaltype2.IT2MF import IT2MF_Triangular, IT2MF_Gaussian
from common.Input import Input
from common.Output import Output
from type1.T1MF import T1MF_Gaussian
from type1.T1MF import T1MF_Triangular
from type2.T2MF import T2MF_Triangular, T2MF_Gaussian
from type2.system.T2_Antecedent import T2_Antecedent
from type2.system.T2_Consequent import T2_Consequent
from type2.system.T2_Rule import T2_Rule
from type2.system.T2_Rulebase import T2_Rulebase


def centerTypeReduction(ruleBase, tip):
    time_start = time.time()
    print("Using center of sets type reduction, the IT2 FLS recommends a tip of:  " + str(ruleBase.evaluate(0).get(tip)))
    time_end = time.time()
    print("the time of centerTypeReduction is " + str(time_end-time_start))

def centroidTypeReduction(ruleBase, tip):
    time_start = time.time()
    print("Using centroid type reduction, the IT2 FLS recommends a tip of:  " + str(ruleBase.evaluate(1).get(tip)))
    time_end = time.time()
    print("the time of centroidTypeReduction is " + str(time_end-time_start))


def exam(foodQuality :float, serviceLevel:float):
    numberOfzLevels = 4
    typeReduction = 0
    xDiscs = 50
    yDiscs = 10

    inputumf = T1MF_Gaussian("inputumf",7,3)
    inputlmf = T1MF_Gaussian("inputlmf", 7, 2)
    inputMfprimer = IT2MF_Gaussian("inputmfprimer",inputumf,inputlmf)
    inputMf = T2MF_Gaussian("inputmf",inputMfprimer,numberOfzLevels)
    food = Input("Food Quality", [0.0, 10.0])
    food.setInputMF(inputMf)
    service = Input("Service Level", [0.0, 10.0])
    tip = Output("Tip", [0.0, 30.0])

    badFoodUMF = T1MF_Triangular("Upper MF for bad food", 0.0, 0.0, 10.0)
    badFoodLMF = T1MF_Triangular("Lower MF for bad food", 0.0, 0.0, 8.0)
    badFoodIT2MF = IT2MF_Triangular("IT2MF for bad food",badFoodUMF,badFoodLMF)
    badFoodMF = T2MF_Triangular("T2MF for bad food", badFoodIT2MF, numberOfzLevels)

    greatFoodUMF = T1MF_Triangular("Upper MF for great food", 0.0, 10.0, 10.0)
    greatFoodLMF = T1MF_Triangular("Lower MF for great food", 2.0, 10.0, 10.0)
    greatFoodIT2MF = IT2MF_Triangular("IT2MF for great food", greatFoodUMF,greatFoodLMF)
    greatFoodMF = T2MF_Triangular("T2MF for great food", greatFoodIT2MF, numberOfzLevels)

    unfriendlyServiceUMF = T1MF_Triangular("Upper MF for unfriendly service", 0.0, 0.0, 8.0)
    unfriendlyServiceLMF = T1MF_Triangular("Lower MF for unfriendly service", 0.0, 0.0, 6.0)
    unfriendlyServiceIT2MF = IT2MF_Triangular("IT2MF for unfriendly service", unfriendlyServiceUMF,unfriendlyServiceLMF)
    unfriendlyServiceMF = T2MF_Triangular("MF for unfriendly service", unfriendlyServiceIT2MF, numberOfzLevels)

    friendlyServiceUMF = T1MF_Triangular("Upper MF for friendly service", 2.0, 10.0, 10.0)
    friendlyServiceLMF = T1MF_Triangular("Lower MF for friendly service", 4.0, 10.0, 10.0)
    friendlyServiceIT2MF = IT2MF_Triangular("IT2MF for friendly service", friendlyServiceUMF,friendlyServiceLMF)
    friendlyServiceMF = T2MF_Triangular("MF for friendly service", friendlyServiceIT2MF, numberOfzLevels)

    lowTipUMF = T1MF_Gaussian("Upper MF for Low tip", 0.0, 6.0)
    lowTipLMF = T1MF_Gaussian("Lower MF for Low tip", 0.0, 4.0)
    lowTipIT2MF = IT2MF_Gaussian("IT2MF for Low tip", lowTipUMF,lowTipLMF)
    lowTipMF = T2MF_Gaussian("MF for Low tip", lowTipIT2MF, numberOfzLevels)

    mediumTipUMF = T1MF_Gaussian("Upper MF for Medium tip", 15.0, 6.0)
    mediumTipLMF = T1MF_Gaussian("Lower MF for Medium tip", 15.0, 4.0)
    mediumTipIT2MF = IT2MF_Gaussian("IT2MF for Medium tip", mediumTipUMF,mediumTipLMF)
    mediumTipMF = T2MF_Gaussian("T2MF for Medium tip", mediumTipIT2MF, numberOfzLevels)

    highTipUMF = T1MF_Gaussian("Upper MF for High tip", 30.0, 6.0)
    highTipLMF = T1MF_Gaussian("Lower MF for High tip", 30.0, 4.0)
    highTipIT2MF = IT2MF_Gaussian("IT2MF for High tip", highTipUMF,highTipLMF)
    highTipMF = T2MF_Gaussian("MF for High tip", highTipIT2MF, numberOfzLevels)

    badFood = T2_Antecedent("BadFood", badFoodMF, food)
    greatFood = T2_Antecedent("GreatFood", greatFoodMF, food)

    unfriendlyService = T2_Antecedent("UnfriendlyService", unfriendlyServiceMF, service)
    friendlyService = T2_Antecedent("FriendlyService", friendlyServiceMF, service)


    lowTip = T2_Consequent("LowTip", lowTipMF, tip)
    mediumTip = T2_Consequent("MediumTip", mediumTipMF, tip)
    highTip = T2_Consequent("HighTip", highTipMF, tip)

    ruleBase = T2_Rulebase(4)
    ruleBase.addrule(T2_Rule([badFood, unfriendlyService], lowTip))
    ruleBase.addrule(T2_Rule([badFood, friendlyService], mediumTip))
    ruleBase.addrule(T2_Rule([greatFood, unfriendlyService], lowTip))
    ruleBase.addrule(T2_Rule([greatFood, friendlyService], highTip))

    tip.setDiscretisationLevel(100)

    food.setInput(foodQuality)
    service.setInput(serviceLevel)

    print("the food is %f , the service is %f\n" %(food.getInput(),service.getInput()))
    # evaluate(0) -> centerTypeReduction

    centerTypeReduction(ruleBase, tip)

    # evaluate(1) -> centroidTypeReduction
    centroidTypeReduction(ruleBase, tip)


if __name__ == '__main__':
    exam(7,8)

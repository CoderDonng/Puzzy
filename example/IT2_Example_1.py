import matplotlib.pyplot as plt
import time
from intervaltype2.IT2MF import IT2MF_Triangular, IT2MF_Gauangle, IT2MF_Gaussian
from intervaltype2.system.IT2_Antecedent import IT2_Antecedent
from intervaltype2.system.IT2_Consequent import IT2_Consequent
from intervaltype2.system.IT2_Rule import IT2_Rule
from intervaltype2.system.IT2_Rulebase import IT2_Rulebase
from common.plot import PlotFun
from common.Input import Input
from common.Output import Output
from type1.T1MF import T1MF_Gauangle
from type1.T1MF import T1MF_Gaussian
from type1.T1MF import T1MF_Triangular

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
    food = Input("Food Quality", [0.0, 10.0])
    service = Input("Service Level", [0.0, 10.0])
    tip = Output("Tip", [0.0, 30.0])

    badFoodUMF = T1MF_Triangular("Upper MF for bad food", 0, 1.0, 10.0)
    badFoodLMF = T1MF_Triangular("Lower MF for bad food", 0.5, 1.0, 8.0)
    badFoodMF = IT2MF_Triangular("IT2MF for bad food",badFoodUMF,badFoodLMF)

    greatFoodUMF = T1MF_Triangular("Upper MF for great food", 0.0, 8.0, 10.0)
    greatFoodLMF = T1MF_Triangular("Lower MF for great food", 2.0, 8.0, 8.0)
    greatFoodMF = IT2MF_Triangular("IT2MF for great food", greatFoodUMF,greatFoodLMF)

    unfriendlyServiceLMF = T1MF_Gauangle("Upper MF for unfriendly service", 0.0, 0.0, 8.0)
    unfriendlyServiceUMF = T1MF_Gauangle("Lower MF for unfriendly service", 0.0, 0.0, 6.0)
    unfriendlyServiceMF = IT2MF_Gauangle("IT2MF for unfriendly service", unfriendlyServiceUMF,unfriendlyServiceLMF)

    # okServiceMF = T1MF_Gauangle("MF for ok service", 2.5, 5.0, 7.5)
    friendlyServiceUMF = T1MF_Gauangle("Upper MF for friendly service", 2.0, 10.0, 10.0)
    friendlyServiceLMF = T1MF_Gauangle("Lower MF for friendly service", 4.0, 10.0, 10.0)
    friendlyServiceMF = IT2MF_Gauangle("IT2MF for friendly service", friendlyServiceUMF,friendlyServiceLMF)

    lowTipUMF = T1MF_Gaussian("Upper MF Low tip", 0.0, 6.0)
    lowTipLMF = T1MF_Gaussian("Lower MF Low tip", 0.0, 4.0)
    lowTipMF = IT2MF_Gaussian("IT2MF for Low tip", lowTipUMF,lowTipLMF)

    mediumTipUMF = T1MF_Gaussian("Upper MF Medium tip", 15.0, 6.0)
    mediumTipLMF = T1MF_Gaussian("Lower MF Medium tip", 15.0, 4.0)
    mediumTipMF = IT2MF_Gaussian("Medium tip", mediumTipUMF,mediumTipLMF)

    highTipUMF = T1MF_Gaussian("Upper MF High tip", 30.0, 6.0)
    highTipLMF = T1MF_Gaussian("Lower MF High tip", 30.0, 4.0)
    highTipMF = IT2MF_Gaussian("IT2MF for High tip", highTipUMF,highTipLMF)


    badFood = IT2_Antecedent("BadFood", badFoodMF, food)
    greatFood = IT2_Antecedent("GreatFood", greatFoodMF, food)

    unfriendlyService = IT2_Antecedent("UnfriendlyService", unfriendlyServiceMF, service)
    # okService = T1_Antecedent("OkService", okServiceMF, service)
    friendlyService = IT2_Antecedent("FriendlyService", friendlyServiceMF, service)

    lowTip = IT2_Consequent("LowTip", lowTipMF, tip, None)
    mediumTip = IT2_Consequent("MediumTip", mediumTipMF, tip,None)
    highTip = IT2_Consequent("HighTip", highTipMF, tip,None)

    ruleBase = IT2_Rulebase()
    ruleBase.addRule(IT2_Rule([badFood, unfriendlyService], lowTip))
    ruleBase.addRule(IT2_Rule([badFood, friendlyService], mediumTip))
    ruleBase.addRule(IT2_Rule([greatFood, unfriendlyService], lowTip))
    ruleBase.addRule(IT2_Rule([greatFood, friendlyService], highTip))

    tip.setDiscretisationLevel(100000)

    food.setInput(foodQuality)
    service.setInput(serviceLevel)

    print("the food is %f , the service is %f\n" %(food.getInput(),service.getInput()))
    # evaluate(0) -> centerTypeReduction

    centerTypeReduction(ruleBase, tip)

    # evaluate(1) -> centroidTypeReduction
    centroidTypeReduction(ruleBase, tip)

    # plot
    plt.figure(1)
    PlotFun().plot_IT2_MFs([badFoodMF, greatFoodMF], 100, food.getDomain(), [0, 1.0], False)
    plt.figure(2)
    PlotFun().plot_IT2_MFs([unfriendlyServiceMF,friendlyServiceMF], 100, service.getDomain(), [0, 1.0], False)
    plt.figure(3)
    PlotFun().plot_IT2_MFs([lowTipMF, mediumTipMF, highTipMF], 100, tip.getDomain(), [0, 1.0], False)
    plt.show()





if __name__ == '__main__':
    exam(7,8)

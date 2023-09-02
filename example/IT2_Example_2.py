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

def centerTypeReduction(ruleBase, tip, smile):
    time_start = time.time()
    print("Using center of sets type reduction, the IT2 FLS recommends a tip of:  " + str(ruleBase.evaluate(0).get(tip))+" and a smile of: "+str(ruleBase.evaluate(0).get(smile)))
    time_end = time.time()
    print("the time of centerTypeReduction is " + str(time_end-time_start))

def centroidTypeReduction(ruleBase, tip, smile):
    time_start = time.time()
    print("Using centroid type reduction, the IT2 FLS recommends a tip of:  " + str(ruleBase.evaluate(1).get(tip))+" and a smile of: "+str(ruleBase.evaluate(1).get(smile)))
    time_end = time.time()
    print("the time of centroidTypeReduction is " + str(time_end-time_start))


def exam(foodQuality :float, serviceLevel:float):
    food = Input("Food Quality", [0.0, 10.0])
    service = Input("Service Level", [0.0, 10.0])
    tip = Output("Tip", [0.0, 30.0])
    smile = Output("Smile", [0, 1])

    badFoodUMF = T1MF_Triangular("Upper MF for bad food", 0.0, 0.0, 10.0)
    badFoodLMF = T1MF_Triangular("Lower MF for bad food", 0.0, 0.0, 8.0)
    badFoodMF = IT2MF_Triangular("IT2MF for bad food",badFoodUMF,badFoodLMF)

    greatFoodUMF = T1MF_Triangular("Upper MF for great food", 0.0, 10.0, 10.0)
    greatFoodLMF = T1MF_Triangular("Lower MF for great food", 2.0, 10.0, 10.0)
    greatFoodMF = IT2MF_Triangular("IT2MF for great food", greatFoodUMF,greatFoodLMF)

    unfriendlyServiceLMF = T1MF_Gauangle("MF for unfriendly service", 0.0, 0.0, 8.0)
    unfriendlyServiceUMF = T1MF_Gauangle("MF for unfriendly service", 0.0, 0.0, 6.0)
    unfriendlyServiceMF = IT2MF_Gauangle("MF for unfriendly service", unfriendlyServiceUMF,unfriendlyServiceLMF)

    # okServiceMF = T1MF_Gauangle("MF for ok service", 2.5, 5.0, 7.5)
    friendlyServiceUMF = T1MF_Gauangle("Upper MF for friendly service", 2.0, 10.0, 10.0)
    friendlyServiceLMF = T1MF_Gauangle("Lower MF for friendly service", 4.0, 10.0, 10.0)
    friendlyServiceMF = IT2MF_Gauangle("MF for friendly service", friendlyServiceUMF,friendlyServiceLMF)

    lowTipUMF = T1MF_Gaussian("Upper MF for Low tip", 0.0, 6.0)
    lowTipLMF = T1MF_Gaussian("Lower MF for Low tip", 0.0, 4.0)
    lowTipMF = IT2MF_Gaussian("IT2MF for Low tip", lowTipUMF,lowTipLMF)

    mediumTipUMF = T1MF_Gaussian("Upper MF for Medium tip", 15.0, 6.0)
    mediumTipLMF = T1MF_Gaussian("Lower MF for Medium tip", 15.0, 4.0)
    mediumTipMF = IT2MF_Gaussian("IT2MF for Medium tip", mediumTipUMF,mediumTipLMF)

    highTipUMF = T1MF_Gaussian("Upper MF for High tip", 30.0, 6.0)
    highTipLMF = T1MF_Gaussian("Lower MF for High tip", 30.0, 4.0)
    highTipMF = IT2MF_Gaussian("IT2MF for High tip", highTipUMF,highTipLMF)

    smallSmileUMF = T1MF_Triangular("Upper MF for Small Smile", 0, 0, 1.0)
    smallSmileLMF = T1MF_Triangular("Lower MF for Small Smile", 0.0, 0.0, 0.8)
    smallSmileMF = IT2MF_Triangular("IT2MF for Small Smile", smallSmileUMF, smallSmileLMF)

    bigSmileUMF = T1MF_Triangular("Upper MF for Big Smile", 0.0, 1.0, 1.0)
    bigSmileLMF = T1MF_Triangular("Lower MF for Big Smile", 0.2, 1.0, 1.0)
    bigSmileMF = IT2MF_Triangular("IT2MF for Big Smile", bigSmileUMF, bigSmileLMF)

    badFood = IT2_Antecedent("BadFood", badFoodMF, food)
    greatFood = IT2_Antecedent("GreatFood", greatFoodMF, food)

    unfriendlyService = IT2_Antecedent("UnfriendlyService", unfriendlyServiceMF, service)
    friendlyService = IT2_Antecedent("FriendlyService", friendlyServiceMF, service)

    lowTip = IT2_Consequent("LowTip", lowTipMF, tip, None)
    mediumTip = IT2_Consequent("MediumTip", mediumTipMF, tip,None)
    highTip = IT2_Consequent("HighTip", highTipMF, tip,None)

    smallSmile = IT2_Consequent("SmallSmile", smallSmileMF, smile, None)
    bigSmile = IT2_Consequent("BigSmile", bigSmileMF, smile, None)

    ruleBase = IT2_Rulebase()
    ruleBase.addRule(IT2_Rule([badFood, unfriendlyService], [lowTip, smallSmile]))
    ruleBase.addRule(IT2_Rule([badFood, friendlyService], mediumTip))
    ruleBase.addRule(IT2_Rule([greatFood, unfriendlyService], lowTip))
    ruleBase.addRule(IT2_Rule([greatFood, friendlyService], [highTip, bigSmile]))

    tip.setDiscretisationLevel(10000)

    food.setInput(foodQuality)
    service.setInput(serviceLevel)

    print("the food is %f , the service is %f\n" %(food.getInput(),service.getInput()))
    # evaluate(0) -> centerTypeReduction

    # centerTypeReduction(ruleBase, tip, smile)

    # evaluate(1) -> centroidTypeReduction
    centroidTypeReduction(ruleBase, tip, smile)

    # plot
    plt.figure(1)
    PlotFun().plot_IT2_MFs([badFoodMF, greatFoodMF], 100, food.getDomain(), [0, 1.0], False)
    plt.figure(2)
    PlotFun().plot_IT2_MFs([unfriendlyServiceMF,friendlyServiceMF], 100, service.getDomain(), [0, 1.0], False)
    plt.figure(3)
    PlotFun().plot_IT2_MFs([lowTipMF, mediumTipMF, highTipMF], 100, tip.getDomain(), [0, 1.0], False)
    plt.figure(4)
    PlotFun().plot_IT2_MFs([smallSmileMF, bigSmileMF], 100, smile.getDomain(), [0, 1.0], False)
    plt.show()


if __name__ == '__main__':
    exam(7,8)

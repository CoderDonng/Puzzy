import matplotlib.pyplot as plt
from common.plot import PlotFun
from common.Input import Input
from common.Output import Output
from type1.T1MF import T1MF_Gauangle
from type1.T1MF import T1MF_Gaussian
from type1.T1MF import T1MF_Triangular
from type1.system.T1_Antecedent import T1_Antecedent
from type1.system.T1_Consequent import T1_Consequent
from type1.system.T1_Rule import T1_Rule
from type1.system.T1_Rulebase import T1_Rulebase



def exam(foodQuality :float, serviceLevel:float):
    food = Input("Food Quality", [0.0, 10.0])
    service = Input("Service Level", [0.0, 10.0])
    tip = Output("Tip", [0.0, 30.0])
    smile = Output("Smile",[0,1])

    badFoodMF = T1MF_Triangular("MF for bad food", 0.0, 0.0, 10.0)
    greatFoodMF = T1MF_Triangular("MF for great food", 0.0, 10.0, 10.0)

    unfriendlyServiceMF = T1MF_Gauangle("MF for unfriendly service", 0.0, 0.0, 6.0)
    okServiceMF = T1MF_Gauangle("MF for ok service", 2.5, 5.0, 7.5)
    friendlyServiceMF = T1MF_Gauangle("MF for friendly service", 4.0, 10.0, 10.0)

    lowTipMF = T1MF_Gaussian("Low tip", 0.0, 6.0)
    mediumTipMF = T1MF_Gaussian("Medium tip", 15.0, 6.0)
    highTipMF = T1MF_Gaussian("High tip", 30.0, 6.0)

    smallSmileMF = T1MF_Triangular("MF for Small Smile", 0.0, 0.0, 1.0)
    bigSmileMF = T1MF_Triangular("MF for Big Smile", 0.0, 1.0, 1.0)


    badFood = T1_Antecedent("BadFood", badFoodMF, food)
    greatFood = T1_Antecedent("GreatFood", greatFoodMF, food)

    unfriendlyService = T1_Antecedent("UnfriendlyService", unfriendlyServiceMF, service)
    okService = T1_Antecedent("OkService", okServiceMF, service)
    friendlyService = T1_Antecedent("FriendlyService", friendlyServiceMF, service)

    lowTip = T1_Consequent("LowTip", lowTipMF, tip)
    mediumTip = T1_Consequent("MediumTip", mediumTipMF, tip)
    highTip = T1_Consequent("HighTip", highTipMF, tip)

    smallSmile = T1_Consequent("SmallSmile", smallSmileMF, smile)
    bigSmile = T1_Consequent("BigSmile", bigSmileMF, smile)

    ruleBase = T1_Rulebase()
    ruleBase.addRule(T1_Rule([[badFood, unfriendlyService]], [lowTip, smallSmile]))
    ruleBase.addRule(T1_Rule([[badFood, okService]], [lowTip, smallSmile]))
    ruleBase.addRule(T1_Rule([[badFood, friendlyService]], [mediumTip]))
    ruleBase.addRule(T1_Rule([[greatFood, unfriendlyService]], [lowTip]))
    ruleBase.addRule(T1_Rule([[greatFood, okService]], [mediumTip,smallSmile]))
    ruleBase.addRule(T1_Rule([[greatFood, friendlyService]], [highTip,bigSmile]))

    tip.setDiscretisationLevel(50)

    food.setInput(foodQuality)
    service.setInput(serviceLevel)

    print("the food is %f , the service is %f\n" %(food.getInput(),service.getInput()))
    # evaluate(0) -> height defuzzification
    print("Using height defuzzification, the FLS recommends a tip of:  " +str(ruleBase.evaluate(0).get(tip))+" and a smile of: "+str(ruleBase.evaluate(0).get(smile)))
    # evaluate(1) -> centroid defuzzification
    print("Using centroid defuzzification, the FLS recommends a tip of:  " + str(ruleBase.evaluate(1).get(tip))+" and a smile of: "+str(ruleBase.evaluate(1).get(smile)))

    # plot
    plt.figure(1)
    PlotFun().plot_T1_MFs([badFoodMF, greatFoodMF], 100, food.getDomain(), [0, 1.0], False)
    plt.figure(2)
    PlotFun().plot_T1_MFs([unfriendlyServiceMF, okServiceMF, friendlyServiceMF], 100, service.getDomain(), [0, 1.0], False)
    plt.figure(3)
    PlotFun().plot_T1_MFs([lowTipMF, mediumTipMF, highTipMF], 100, tip.getDomain(), [0, 1.0], False)
    plt.figure(4)
    PlotFun().plot_T1_MFs([smallSmileMF, bigSmileMF], 100, smile.getDomain(), [0, 1.0], False)
    plt.show()





if __name__ == '__main__':
    exam(7,8)


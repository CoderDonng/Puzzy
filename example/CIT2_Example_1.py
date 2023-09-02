
from datashape import Null
from matplotlib import pyplot as plt
from CIT2.CIT2 import CIT2
from CIT2.Generator import Generator_Triangular, Generator_Gauangle, Generator_Gaussian
from CIT2.system.CIT2_Antecedent import CIT2_Antecedent
from CIT2.system.CIT2_Consequent import CIT2_Consequent
from CIT2.system.CIT2_Rule import CIT2_Rule
from CIT2.system.CIT2_Rulebase import CIT2_Rulebase
from common.Input import Input
from common.Output import Output
from common.plot import PlotFun


def exam(foodQuality :float, serviceLevel:float):
    shifting_size_1 = 1
    shifting_size_2 = 0.5
    food = Input("Food Quality", [0.0, 10.0])
    service = Input("Service Level", [0.0, 10.0])
    tip = Output("Tip", [0.0, 30.0])

    badFoodMF = Generator_Triangular("Bad", 0.0, 0.0, 10.0)
    badFoodMF.setleftShoulder(True)
    greatFoodMF = Generator_Triangular("Great", 0.0, 10.0, 10.0)
    greatFoodMF.setrightShoulder(True)
    unfriendlyServiceMF = Generator_Gauangle("Unfriendly", 0.0, 0.0, 6.0)
    unfriendlyServiceMF.setleftShoulder(True)
    okServiceMF = Generator_Gauangle("OK", 2.5, 5.0, 7.5)
    friendlyServiceMF = Generator_Gauangle("Friendly", 4.0, 10.0, 10.0)
    friendlyServiceMF.setrightShoulder(True)
    lowTipMF = Generator_Gaussian("Low", 0.0, 6.0)
    lowTipMF.setleftShoulder(True)
    mediumTipMF = Generator_Gaussian("Medium", 15.0, 6.0)
    highTipMF = Generator_Gaussian("High", 30.0, 6.0)
    highTipMF.setrightShoulder(True)

    cit2_badFoodMF = CIT2(badFoodMF.getName(), badFoodMF, shifting_size_2)
    cit2_greatFoodMF = CIT2(greatFoodMF.getName(), greatFoodMF, shifting_size_2)

    cit2_unfriendlyServiceMF = CIT2(unfriendlyServiceMF.getName(), unfriendlyServiceMF, shifting_size_2)
    cit2_okServiceMF = CIT2(okServiceMF.getName(), okServiceMF, shifting_size_2)
    cit2_friendlyServiceMF = CIT2(friendlyServiceMF.getName(), friendlyServiceMF, shifting_size_2)

    cit2_lowTipMF = CIT2(lowTipMF.getName(), lowTipMF, shifting_size_1)
    cit2_mediumTipMF = CIT2(mediumTipMF.getName(), mediumTipMF, shifting_size_1)
    cit2_highTipMF = CIT2(highTipMF.getName(), highTipMF, shifting_size_1)




    badFood = CIT2_Antecedent(cit2_badFoodMF, food)
    greatFood = CIT2_Antecedent(cit2_greatFoodMF, food)

    unfriendlyService = CIT2_Antecedent(cit2_unfriendlyServiceMF, service)
    okService = CIT2_Antecedent(cit2_okServiceMF, service)
    friendlyService = CIT2_Antecedent(cit2_friendlyServiceMF, service)

    lowTip = CIT2_Consequent(cit2_lowTipMF, tip)
    mediumTip = CIT2_Consequent(cit2_mediumTipMF, tip)
    highTip = CIT2_Consequent(cit2_highTipMF, tip)

    ruleBase = CIT2_Rulebase(Null)
    ruleBase.addRule(CIT2_Rule([badFood], lowTip))
    ruleBase.addRule(CIT2_Rule([greatFood, unfriendlyService], lowTip))
    ruleBase.addRule(CIT2_Rule([greatFood, okService], mediumTip))
    ruleBase.addRule(CIT2_Rule([greatFood, friendlyService], highTip))


    food.setInput(foodQuality)
    service.setInput(serviceLevel)

    print("the food is %f , the service is %f\n" %(food.getInput(),service.getInput()))
    result = ruleBase.explainableDefuzzification(1000)
    print("The recommended tip percentage is in the range: "+str(result.getConstrainedCentroid()))


    # plot
    plt.figure(1)
    PlotFun().plot_IT2_MFs([cit2_badFoodMF, cit2_greatFoodMF], 100, food.getDomain(), [0, 1.0], False)
    plt.figure(2)
    PlotFun().plot_IT2_MFs([cit2_unfriendlyServiceMF,cit2_friendlyServiceMF], 100, service.getDomain(), [0, 1.0], False)
    plt.figure(3)
    PlotFun().plot_IT2_MFs([cit2_lowTipMF, cit2_mediumTipMF, cit2_highTipMF], 100, tip.getDomain(), [0, 1.0], False)
    plt.show()





if __name__ == '__main__':
    exam(7,2)

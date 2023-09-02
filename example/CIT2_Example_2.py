
from datashape import Null
from matplotlib import pyplot as plt
from CIT2.CIT2 import CIT2
from CIT2.Gen_CIT2 import Gen_CIT2
from CIT2.Generator import Generator_Triangular, Generator_Gauangle, Generator_Gaussian
from CIT2.system.CIT2_Antecedent import CIT2_Antecedent
from CIT2.system.CIT2_Consequent import CIT2_Consequent
from CIT2.system.CIT2_Rule import CIT2_Rule
from CIT2.system.CIT2_Rulebase import CIT2_Rulebase
from common.Input import Input
from common.Output import Output
from common.plot import PlotFun


def exam(foodQuality :float, serviceLevel:float):
    food = Input("Food Quality", [0.0, 10.0])
    service = Input("Service Level", [0.0, 10.0])
    tip = Output("Tip", [0.0, 30.0])

    cit2_badFoodMF = Gen_CIT2("Bad", "T1MF_Triangular", [[-0.5,0.5], [-0.5,0.5], [9.5,10.5]])
    cit2_greatFoodMF = Gen_CIT2("Great", "T1MF_Triangular", [[-0.5,0.5], [9.5,10.5], [9.5,10.5]])

    cit2_unfriendlyServiceMF = Gen_CIT2("Unfriendly", "T1MF_Gauangle", [[-0.5,0.5], [-0.5,0.5], [5.5,6.5]])
    cit2_okServiceMF = Gen_CIT2("OK", "T1MF_Gauangle", [[2,3], [4.5,5.5], [7,8]])
    cit2_friendlyServiceMF = Gen_CIT2("Friendly", "T1MF_Gauangle", [[3.5, 4.5], [9.5,10.5], [9.5,10.5]])

    cit2_lowTipMF = Gen_CIT2("Low", "T1MF_Gaussian", [[-1, 1], 6])
    cit2_mediumTipMF = Gen_CIT2("Medium", "T1MF_Gaussian", [[14, 16], 6])
    cit2_highTipMF = Gen_CIT2("High", "T1MF_Gaussian", [[29, 31], 6])




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
    result = ruleBase.explainableDefuzzification(50)
    print("M1: The recommended tip percentage is in the range: "+str(result.getConstrainedCentroid()))

    plt.figure(1)
    PlotFun().plot_IT2_MFs([cit2_badFoodMF, cit2_greatFoodMF], 100, food.getDomain(), [0, 1.0], False)
    plt.figure(2)
    PlotFun().plot_IT2_MFs([cit2_unfriendlyServiceMF,cit2_okServiceMF,cit2_friendlyServiceMF], 100, service.getDomain(), [0, 1.0], False)
    plt.figure(3)
    PlotFun().plot_IT2_MFs([cit2_lowTipMF, cit2_mediumTipMF, cit2_highTipMF], 100, tip.getDomain(), [0, 1.0], False)
    plt.show()





if __name__ == '__main__':
    exam(7,2)

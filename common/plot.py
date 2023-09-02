import random
import string

import numpy
import matplotlib.pyplot as plt

class PlotFun:

    def discrete(self, support:[float,float], discLevel:int):
        d = numpy.zeros(discLevel).tolist()
        stepSize = (support[1] - support[0])/(discLevel-1.0)
        d[0] = support[0]
        d[len(d)-1] = support[1]
        for i in range(1,len(d)):
            d[i] = support[0]+i*stepSize

        return d



    def plot_T1_MFs(self, sets, xDisc, xAxisRange, yAxisRange, addExtraEndpoints):
        # t = "".join(random.sample(string.ascii_letters, 3))
        for k in range(len(sets)):
            x = self.discrete(sets[k].getSupport(),xDisc)
            y = []
            for i in range(xDisc):
                y.append(sets[k].getFS(x[i]))

            if addExtraEndpoints:
                x_2 = numpy.zeros(len(x)+2).tolist()
                y_2 = numpy.zeros(len(y)+2).tolist()

                x_2[0] = sets[k].getSupport()[0]
                x_2[len(x_2)-1] = sets[k].getSupport()[1]
                y_2[0] = 0.0
                y_2[len(y_2) - 1] = 0.0
                for i in range(len(x)):
                    x_2[i+1] = x[i]
                    y_2[i+1] = y[i]
                x = x_2
                y = y_2

            plt.plot(x,y)
        plt.xlim(xAxisRange[0], xAxisRange[1])
        plt.ylim(yAxisRange[0], yAxisRange[1])
        plt.legend([set.getName() for set in sets])
        # plt.show()

    def plot_IT2_MFs(self, sets, xDisc, xAxisRange, yAxisRange, addExtraEndpoints):
        # t = "".join(random.sample(string.ascii_letters, 3))
        x= []
        y_u = [];y_l = []
        for k in range(len(sets)):
            x = self.discrete(sets[k].getSupport(),xDisc)
            y_u = []; y_l= []
            for i in range(xDisc):
                y_u.append(sets[k].getFS(x[i])[1])
                y_l.append(sets[k].getFS(x[i])[0])

            if addExtraEndpoints:
                x_2 = numpy.zeros(len(x)+2).tolist()
                y_u_2 = numpy.zeros(len(y_u)+2).tolist()
                y_l_2 = numpy.zeros(len(y_l) + 2).tolist()

                x_2[0] = sets[k].getSupport()[0]
                x_2[len(x_2)-1] = sets[k].getSupport()[1]
                y_u_2[0] = 0.0
                y_u_2[len(y_u) - 1] = 0.0
                y_l_2[0] = 0.0
                y_l_2[len(y_l) - 1] = 0.0
                for i in range(len(x)):
                    x_2[i+1] = x[i]
                    y_u_2[i+1] = y_u[i]
                    y_l_2[i + 1] = y_l[i]
                x = x_2
                y_u = y_u_2
                y_l = y_l_2

            plt.fill_between(x,y_l,y_u)
        plt.xlim(xAxisRange[0], xAxisRange[1])
        plt.ylim(yAxisRange[0], yAxisRange[1])

        # plt.show()





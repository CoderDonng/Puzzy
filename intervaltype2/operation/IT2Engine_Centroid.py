import math

from datashape import Null

from intervaltype2.operation.IT2MF_Intersection import IT2MF_Intersection
from intervaltype2.operation.IT2MF_Union import IT2MF_Union


class IT2Engine_Centroid:
    __centroid:[float,float]
    __primaryDiscretizationLevel = 100
    __KM = 0
    __EKM = 1
    __EKM_L0 = 2.4
    __EKM_R0 = 1.7
    __centroid_algorithm_selector = __KM
    __log = False # log for comparison

    def __init__(self, primaryDiscretizationLevel:int):
        self.__primaryDiscretizationLevel = primaryDiscretizationLevel

    def getPrimaryDiscretizationLevel(self):
        return self.__primaryDiscretizationLevel

    def setPrimaryDiscretizationLevel(self,primaryDiscretizationLevel):
        self.__primaryDiscretizationLevel = primaryDiscretizationLevel

    def getCentroid(self, mf):
        if self.__centroid_algorithm_selector == self.__KM:
            self.__centroid = self.getCentroidKM(mf)
        if self.__centroid_algorithm_selector == self.__EKM:
            self.__centroid = self.getCentroidEKM(mf,self.__EKM_L0,self.__EKM_R0)  #为何L0=2.4，R0=1.7

        return self.__centroid

    def getCentroidKM(self, mf):
        if mf == Null or mf is None:
            return [float(math.nan),float(math.nan)]
        if isinstance(mf, IT2MF_Intersection) and not mf.intersectionExists():
            return [float(math.nan),float(math.nan)]

        w:list = []
        x:list = []
        weights:list = []
        y:float
        yDash:float
        y_l = 0
        y_r = 0
        domainSize:float
        temp:float
        k:int
        stopFlag = False
        iterationCounterLeft = 0
        iterationCounterRight = 0

        if isinstance(mf,IT2MF_Union) and mf.isNull(): return None

        if mf.getSupport()[0] == mf.getSupport()[1]: return mf.getSupport()

        domainSize = mf.getSupport()[1] - mf.getSupport()[0]
        temp = domainSize/(self.__primaryDiscretizationLevel-1)
        for i in range(self.__primaryDiscretizationLevel):
            x.append(i*temp+mf.getSupport()[0])
            w.append(mf.getFS(x[i]))
            weights.append((w[i][0]+w[i][1])/2)

        for r in range(2): #循环两次，分别计算l和r
            stopFlag = False
            for i in range(self.__primaryDiscretizationLevel):
                w[i] = mf.getFS(x[i]) #w: [[float, float]], 每个元素由lower mf和upper mf的值组成
                weights[i] = (w[i][0] + w[i][1]) / 2
            y = self.getWeightedSum(x, weights)
            while not stopFlag:
                if self.__log:
                    if r == 0: # 第一次遍历计算l的值
                        iterationCounterLeft += 1
                    else:
                        iterationCounterRight += 1
                k = 0
                while k < self.__primaryDiscretizationLevel-1:
                    if x[k] <= y <= x[k+1]:
                        break
                    if k == self.__primaryDiscretizationLevel-2:
                        print("###  NO k WAS  FOUND! ### for: "+mf.getName()+"\n")
                    k +=1

                for j in range(k+1):
                    weights[j] = w[j][1-r]
                for j in range(k+1,self.__primaryDiscretizationLevel):
                    weights[j] = w[j][r]

                yDash = self.getWeightedSum(x, weights)
                if math.isnan(yDash):
                    yDash = y
                if abs(yDash-y)<0.001:
                    stopFlag = True
                    if r == 0:
                        y_l = yDash
                    else:
                        y_r = yDash
                else:
                    y = yDash

        return [y_l,y_r]

    def getCentroidEKM(self,mf, divisor_left, divisor_right):
        w: list = []; x: list = []
        y: float = 0; yDash: float = 0; y_l: float = 0; y_r: float = 0
        domainSize: float
        k: int; kDash:int
        aDash = 0.0; bDash = 0.0; s = 0.0
        stopFlag = False
        iterationCounterLeft = 0; iterationCounterRight = 0
        log = True
        domainSize = mf.getSupport()[1] - mf.getSupport()[0]
        temp = float(domainSize/self.__primaryDiscretizationLevel)
        for i in range(self.__primaryDiscretizationLevel+1):
            x.append(i*temp+mf.getSupport()[0])
            w.append(mf.getFS(x[i]))
        for r in range(2): # r = 0, left calculation; r = 1, left calculation

            stopFlag =False
            if r ==0:
                k = int(round(self.__primaryDiscretizationLevel/divisor_left)) #通过对divisor_left和divisor_right随机取值获取初始k值，减少遍历次数
            else:
                k = int(round(self.__primaryDiscretizationLevel / divisor_right))
            a = 0.0; b = 0.0
            for m in range(k+1):
                a += x[m]*w[m][1-r]
                b += w[m][1-r]
            for m in range(k+1,self.__primaryDiscretizationLevel):
                a += x[m] * w[m][r]
                b += w[m][r]
            y = a/b #这里的做法与KM类似， 以K值为界选取山下隶属度函数值作为权重

            while not stopFlag:
                if log:
                    if r == 0:
                        iterationCounterLeft += 1 #记录计算l的迭代次数
                    else:
                        iterationCounterRight += 1 #记录计算r的迭代次数
                kDash = 0
                while kDash < self.__primaryDiscretizationLevel:
                    if x[kDash] <= y <= x[kDash+1]:
                        break
                    kDash += 1
                if kDash == k: #若计算centroid过程中的k值恰好使得centroid落在x[k]和x[k+1]之间，则跳出循环
                    stopFlag = True
                    if r == 0:
                        y_l = y
                    else:
                        y_r = y
                else:
                    s = kDash - k
                    if s < 0:
                        s = -1
                    else:
                        s = 1
                    t = min(k, kDash)+1
                    while t <= max(k,kDash):
                        aDash += x[t]*(w[t][1]-w[t][0])
                        bDash += w[t][1]-w[t][0]
                        t += 1
                    if r == 0:
                        aDash = a + s * aDash #计算l时，若k过小，则增加x*w[1]，减少x*w[0]；过大则增加x*w[0]，减少x*w[1]，运算方向由s决定
                        bDash = b + s * bDash
                    else:
                        aDash = a - s * aDash #同理
                        bDash = b - s * bDash

                    yDash = aDash/bDash
                    y = yDash
                    a = aDash
                    b = bDash
                    k = kDash
                    aDash = 0; bDash = 0

        return [y_l,y_r]

    def getWeightedSum(self, x:list, w:list):
        temp = 0.0
        temp_2 = 0.0
        for i in range(len(x)):
            temp += x[i]*w[i]
            temp_2 += w[i]
        if temp_2 != 0:
            return temp/temp_2
        else:
            return float(math.nan)
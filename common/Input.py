import numpy

from intervaltype2.IT2MF import IT2MF_Gaussian, IT2MF_Gauangle, IT2MF_Triangular, IT2MF_Trapezoidal
from type1.T1MF import T1MF_Gaussian, T1MF_Gauangle, T1MF_Triangular, T1MF_Trapezoidal, T1MF_Singleton
from type2.T2MF import T2MF_Gaussian, T2MF_Triangular, T2MF_Trapezoidal


class Input:
    x: float
    __name: str
    __domain: [float, float]

    def __init__(self, name: str, domain: [float, float]):
        self.__name = name
        self.__domain = domain
        self.x = 0
        self.__inputMF = T1MF_Singleton("",self.x)


    def getInput(self):
        return self.x

    def setInput(self, x:float):
        if self.__domain[0]<=x<=self.__domain[1]:
            self.x = x
            inMF = self.__inputMF
            nameMF = inMF.name

            if isinstance(inMF, T1MF_Singleton):
                self.__inputMF = T1MF_Singleton("",x)
            elif isinstance(inMF, T1MF_Gaussian):
                spread = inMF.getSpread()
                self.__inputMF = T1MF_Gaussian(nameMF,x,spread)
            elif isinstance(inMF, T1MF_Gauangle):
                start = inMF.getStart()
                end = inMF.getEnd()
                mean = inMF.getMean()
                self.__inputMF = T1MF_Gauangle(nameMF,start+(x-mean),x, end+(x-mean))
            elif isinstance(inMF, T1MF_Triangular):
                start = inMF.getStart()
                end = inMF.getEnd()
                mean = inMF.getPeak()
                self.__inputMF = T1MF_Triangular(nameMF,start+(x-mean),x,end+(x-mean))
            elif isinstance(inMF, T1MF_Trapezoidal):
                params = [0,0,0,0]
                params[0] = inMF.getA()
                params[1] = inMF.getB()
                params[2] = inMF.getC()
                params[3] = inMF.getD()
                mid = (params[1]+params[2])/2
                d = x - mid
                params[0] += d
                params[1] += d
                params[2] += d
                params[3] += d
                self.__inputMF = T1MF_Trapezoidal(nameMF,params)
            elif isinstance(inMF, IT2MF_Gaussian):
                lmf = inMF.getLMF()
                l_name = lmf.getName()
                l_spread = lmf.getSpread()
                umf = inMF.getUMF()
                u_name = umf.getName()
                u_spread = umf.getSpread()
                self.__inputMF = IT2MF_Gaussian(nameMF, T1MF_Gaussian(u_name,x,u_spread), T1MF_Gaussian(l_name,x,l_spread))
            elif isinstance(inMF, IT2MF_Gauangle):
                lmf = inMF.getLMF()
                l_name = lmf.getName()
                l_start = lmf.getStart()
                l_end = lmf.getEnd()
                l_mean = lmf.getMean()
                umf = inMF.getUMF()
                u_name = umf.getName()
                u_start = umf.getStart()
                u_end = umf.getEnd()
                u_mean = umf.getMean()
                self.__inputMF = IT2MF_Gauangle(nameMF,T1MF_Gauangle(u_name,u_start+(x-u_mean),x,u_end+(x-u_mean)),T1MF_Gauangle(l_name,l_start+(x-l_mean),x,l_end+(x-l_mean)))
            elif isinstance(inMF, IT2MF_Triangular):
                lmf = inMF.getLMF()
                l_name = lmf.getName()
                l_start = lmf.getStart()
                l_end = lmf.getEnd()
                l_mean = lmf.getPeak()
                umf = inMF.getUMF()
                u_name = umf.getName()
                u_start = umf.getStart()
                u_end = umf.getEnd()
                u_mean = umf.getPeak()
                self.__inputMF = IT2MF_Triangular(nameMF,T1MF_Triangular(u_name,u_start+(x-u_mean),x,u_end+(x-u_mean)),T1MF_Triangular(l_name,l_start+(x-l_mean),x,l_end+(x-l_mean)))
            elif isinstance(inMF, IT2MF_Trapezoidal):
                l_params = [0,0,0,0]
                u_params = [0,0,0,0]
                lmf = inMF.getLMF()
                l_params[0] = lmf.getA()
                l_params[1] = lmf.getB()
                l_params[2] = lmf.getC()
                l_params[3] = lmf.getD()
                l_mid = (l_params[1]+l_params[2])/2
                l_d = x - l_mid
                l_params[0] += l_d
                l_params[1] += l_d
                l_params[2] += l_d
                l_params[3] += l_d
                LMF = T1MF_Trapezoidal(lmf.getName(), l_params)
                umf = inMF.getUMF()
                u_params[0] = umf.getA()
                u_params[1] = umf.getB()
                u_params[2] = umf.getC()
                u_params[3] = umf.getD()
                u_mid = (u_params[1] + u_params[2]) / 2
                u_d = x - u_mid
                u_params[0] += u_d
                u_params[1] += u_d
                u_params[2] += u_d
                u_params[3] += u_d
                UMF = T1MF_Trapezoidal(umf.getName(), u_params)
                self.__inputMF = IT2MF_Trapezoidal(nameMF, UMF, LMF)
            elif isinstance(inMF, T2MF_Gaussian):
                num = inMF.getNumberOfSlices()
                IT2s = []
                for i in range(num):
                    temp = inMF.getZSlice(i)
                    lmf = temp.getLMF()
                    l_name = lmf.getName()
                    l_spread = lmf.getSpread()
                    umf = temp.getUMF()
                    u_name = umf.getName()
                    u_spread = umf.getSpread()
                    temp = IT2MF_Gaussian(nameMF, T1MF_Gaussian(u_name, x, u_spread), T1MF_Gaussian(l_name, x, l_spread))
                    IT2s.append(temp)

                self.__inputMF = T2MF_Gaussian(nameMF,IT2s, None)
            elif isinstance(inMF, T2MF_Triangular):
                num = inMF.getNumberOfSlices()
                IT2s = []
                for i in range(num):
                    temp = inMF.getZSlice(i)
                    lmf = temp.getLMF()
                    l_name = lmf.getName()
                    l_start = lmf.getStart()
                    l_mean = lmf.getPeak()
                    l_end = lmf.getEnd()
                    umf = temp.getUMF()
                    u_name = umf.getName()
                    u_start = umf.getStart()
                    u_mean = umf.getPeak()
                    u_end = umf.getEnd()
                    temp = IT2MF_Triangular(nameMF, T1MF_Triangular(u_name,u_start+(x-u_mean),x,u_end+(x-u_mean)),T1MF_Triangular(l_name,l_start+(x-l_mean),x,l_end+(x-l_mean)))
                    IT2s.append(temp)

                self.__inputMF = T2MF_Triangular(nameMF, IT2s, None)

            elif isinstance(inMF, T2MF_Trapezoidal):
                num = inMF.getNumberOfSlices()
                IT2s = []
                for i in range(num):
                    temp = inMF.getZSlice(i)
                    params = numpy.zeros(4).tolist()
                    lmf = temp.getLMF()
                    params[0] = lmf.getA()
                    params[1] = lmf.getB()
                    params[2] = lmf.getC()
                    params[3] = lmf.getD()
                    mid = (params[1]+params[2])/2
                    d = x - mid
                    params[0] += d
                    params[1] += d
                    params[2] += d
                    params[3] += d
                    LMF = T1MF_Trapezoidal(lmf.getName(), params)
                    umf = temp.getUMF()
                    params[0] = umf.getA()
                    params[1] = umf.getB()
                    params[2] = umf.getC()
                    params[3] = umf.getD()
                    mid = (params[1] + params[2]) / 2
                    d = x - mid
                    params[0] += d
                    params[1] += d
                    params[2] += d
                    params[3] += d
                    UMF = T1MF_Trapezoidal(umf.getName(), params)
                    temp = IT2MF_Trapezoidal(nameMF, UMF, LMF)
                    IT2s.append(temp)

                self.__inputMF = T2MF_Trapezoidal(nameMF, IT2s,None)
            else:
                raise Exception("The input value "+str(x)+" was rejected "
                + "as it is outside of the domain for this input: "
                + "["+str(self.__domain[0])+", "+str(self.__domain[1])+"].")

    # def getInputMF(self): return self.__inputMF

    def getInputMF(self):
        return self.__inputMF

    def setInputMF(self, inputMF):
        if self.__domain[0]<=inputMF.getPeak()<=self.__domain[1]:
            self.x = inputMF.getPeak()
            self.__inputMF = inputMF
        else:
            raise Exception("The input value "+str(self.x)+" was rejected "
                    + "as it is outside of the domain for this input: "
                    + "["+str(self.__domain[0])+", "+str(self.__domain[1])+"].")

    def getDomain(self): return self.__domain

    def setDomain(self, domain): self.__domain = domain

    def getName(self): return self.__name

    def toString(self):

        return "Input: '"+self.__name+"' with value: "+str(self.x)


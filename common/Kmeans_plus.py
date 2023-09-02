import random
import numpy as np


class Centroids:
    def __init__(self, rootdata, centroidsNum):
        if centroidsNum == 1:
            raise Exception("It's meaningless to classify the data into 1 class.")
        self.__rootdata = rootdata
        self.__centroidsNum = centroidsNum

    # def getCentroids(self):
    #     #初始化
    #     centroids = []
    #     centroids.append(random.choice(self.__rootdata[:,0]))
    #     for i in range(1, self.__centroidsNum):
    #         D = np.power((self.__rootdata - np.array(centroids).reshape(1,len(centroids))),2)
    #         D_min = D.min(1)
    #         d = D_min/sum(D_min)
    #         r = random.random()
    #         centroids.append(self.__rootdata[np.where(np.cumsum(d)>r)[0][0],0])
    #     centroids.sort()
    #
    #     distance = np.zeros((self.__rootdata.shape[0],self.__centroidsNum))
    #     while 1:
    #         mat = [[] for i in range(self.__centroidsNum)]
    #         for i in range(self.__centroidsNum):
    #             for j in range(self.__rootdata.shape[0]):
    #                 distance[j, i] = np.power(self.__rootdata - centroids[i], 2)[j,0]
    #
    #         for i in range(self.__rootdata.shape[0]):
    #             l = distance[i,:]
    #             indexs = np.where(l == min(l))[0]
    #             for it in iter(indexs):
    #                 mat[it].append(self.__rootdata[i,0])
    #
    #         newCentroids = [np.mean(Iter) for Iter in iter(mat)]
    #         if centroids == newCentroids:
    #             stds = [np.std(Iter) for Iter in iter(mat)]
    #             break
    #         else:
    #             centroids = newCentroids
    #
    #     return [centroids, stds]

    def getCentroids(self):
        Centroids = []
        for n in range(5):
            centroids = []
            centroids.append(random.choice(self.__rootdata[:,0]))
            for i in range(1, self.__centroidsNum):
                D = np.power((self.__rootdata - np.array(centroids).reshape(1,len(centroids))),2)
                D_min = D.min(1)
                d = D_min/sum(D_min)
                r = random.random()
                centroids.append(self.__rootdata[np.where(np.cumsum(d)>r)[0][0],0])
            centroids.sort()

            distance = np.zeros((self.__rootdata.shape[0],self.__centroidsNum))
            while 1:
                mat = [[] for i in range(self.__centroidsNum)]
                for i in range(self.__centroidsNum):
                    for j in range(self.__rootdata.shape[0]):
                        distance[j, i] = np.power(self.__rootdata - centroids[i], 2)[j,0]

                for i in range(self.__rootdata.shape[0]):
                    l = distance[i,:]
                    indexs = np.where(l == min(l))[0]
                    for it in iter(indexs):
                        mat[it].append(self.__rootdata[i,0])

                newCentroids = [np.mean(Iter) for Iter in iter(mat)]
                if centroids == newCentroids:
                    break
                else:
                    centroids = newCentroids
            Centroids.append(centroids)

        Centroids = np.array(Centroids)

        return [Centroids.mean(axis=0), Centroids.std(axis=0)]







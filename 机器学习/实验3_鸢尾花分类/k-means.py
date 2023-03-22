import numpy as np
import pandas as pd
import matplotlib.pylab as plt

def LoadData():
    data = pd.read_csv("iris.csv")
    # 获取所有列，并存入一个数组中
    ori_data = np.array(data)
    # 删去数组中的序号列和分类结果列
    delete_index = [0,5]
    fdata = np.delete(ori_data,delete_index,axis=1)
    # 返回最后处理好的数据集
    return fdata

# 欧氏距离计算
def distEclud(x, y):
    return np.sqrt(np.sum((x - y) ** 2))  # 计算欧氏距离

# 为给定数据集构建一个包含K个随机质心centroids的集合
def randCent(dataSet, k):
    m, n = dataSet.shape  # m=150,n=4
    centroids = np.zeros((k, n))  # 4*4
    for i in range(k):  # 执行四次
        index = int(np.random.uniform(0, m))  # 产生0到150的随机数（在数据集中随机挑一个向量做为质心的初值）
        centroids[i, :] = dataSet[index, :]  # 把对应行的四个维度传给质心的集合
    return centroids

# k均值聚类算法
def KMeans(dataSet, k):
    m = np.shape(dataSet)[0]  # 行数150
    # 第一列存每个样本属于哪一簇(四个簇)
    # 第二列存每个样本的到簇的中心点的误差
    clusterAssment = np.mat(np.zeros((m, 2)))  # .mat()创建150*2的矩阵
    clusterChange = True
    # 1.初始化质心centroids
    centroids = randCent(dataSet, k)  # 4*4
    while clusterChange:
        # 样本所属簇不再更新时停止迭代
        clusterChange = False
        # 遍历所有的样本（行数150）
        for i in range(m):
            minDist = 100000.0
            minIndex = -1
            # 遍历所有的质心
            # 2.找出最近的质心
            for j in range(k):
                # 计算该样本到4个质心的欧式距离，找到距离最近的那个质心minIndex
                distance = distEclud(centroids[j, :], dataSet[i, :])
                if distance < minDist:
                    minDist = distance
                    minIndex = j
            # 3.更新该行样本所属的簇
            if clusterAssment[i, 0] != minIndex:
                clusterChange = True
                clusterAssment[i, :] = minIndex, minDist ** 2
        # 4.更新质心
        for j in range(k):
            # np.nonzero(x)返回值不为零的元素的下标，它的返回值是一个长度为x.ndim(x的轴数)的元组
            # 元组的每个元素都是一个整数数组，其值为非零元素的下标在对应轴上的值。
            # 矩阵名.A 代表将 矩阵转化为array数组类型

            # 这里取矩阵clusterAssment所有行的第一列，转为一个array数组，与j（簇类标签值）比较，返回true or false
            # 通过np.nonzero产生一个array，其中是对应簇类所有的点的下标值（x个）
            # 再用这些下标值求出dataSet数据集中的对应行，保存为pointsInCluster（x*4）
            pointsInCluster = dataSet[np.nonzero(clusterAssment[:, 0].A == j)[0]]  # 获取对应簇类所有的点（x*4）
            centroids[j, :] = np.mean(pointsInCluster, axis=0)  # 求均值，产生新的质心
            # axis=0，那么输出是1行4列，求的是pointsInCluster每一列的平均值，即axis是几，那就表明哪一维度被压缩成1
    print("cluster complete")
    return centroids, clusterAssment

def kMeansSSE(dataSet,k):
    m = np.shape(dataSet)[0]
    #分配样本到最近的簇：存[簇序号,距离的平方]
    clusterAssment=np.mat(np.zeros((m,2)))
    #step1:#初始化聚类中心
    centroids = randCent(dataSet, k)
    print('initial centroids=',centroids)
    sseOld=0
    sseNew=np.inf
    iterTime=0 #查看迭代次数
    while(abs(sseNew-sseOld)>0.0001):
        sseOld=sseNew
        #step2:将样本分配到最近的质心对应的簇中
        for i in range(m):
            minDist=100000.0;minIndex=-1
            for j in range(k):
                #计算第i个样本与第j个质心之间的距离
                distJI=distEclud(centroids[j,:],dataSet[i,:])
                #获取到第i样本最近的质心的距离,及对应簇序号
                if distJI<minDist:
                    minDist=distJI;minIndex=j
            clusterAssment[i,:]=minIndex,minDist**2 #分配样本到最近的簇
        iterTime+=1
        sseNew=sum(clusterAssment[:,1])
        print('the SSE of %d'%iterTime + 'th iteration is %f'%sseNew)
        #step3:更新聚类中心
        for cent in range(k):
            #样本分配结束后，重新计算聚类中心
            ptsInClust=dataSet[np.nonzero(clusterAssment[:,0].A==cent)[0]]
            #按列取平均,mean()对array类型
            centroids[cent,:] = np.mean(ptsInClust, axis=0)
    return centroids, clusterAssment,sseNew

def draw_k2(data, center, assment):
    # 取前两个维度（萼片长度、萼片宽度），绘制数据分布图
    plt.scatter(fdata[:, 0], fdata[:, 1], c="red", marker='o', label='see')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.legend(loc=2)
    plt.show()

    length = len(center)
    fig = plt.figure
    data1 = data[np.nonzero(assment[:, 0].A == 0)[0]]
    data2 = data[np.nonzero(assment[:, 0].A == 1)[0]]
    # 选取前两个维度绘制原始数据的散点图
    plt.scatter(data1[:, 0], data1[:, 1], c="red", marker='o', label='label0')
    plt.scatter(data2[:, 0], data2[:, 1], c="green", marker='*', label='label1')
    # 绘制簇的质心点
    for i in range(length):
        plt.annotate('center', xy=(center[i, 0], center[i, 1]), xytext= \
            (center[i, 0] + 1, center[i, 1] + 1), arrowprops=dict(facecolor='yellow'))
        #  plt.annotate('center',xy=(center[i,0],center[i,1]),xytext=\
        # (center[i,0]+1,center[i,1]+1),arrowprops=dict(facecolor='red'))
    plt.show()

    # 取后两个维度（萼片长度、萼片宽度），绘制数据分布图
    plt.scatter(fdata[:, 2], fdata[:, 3], c="red", marker='o', label='see')
    plt.xlabel('petal length')
    plt.ylabel('petal width')
    plt.legend(loc=2)
    plt.show()
    # 选取后两个维度绘制原始数据的散点图
    plt.scatter(data1[:, 2], data1[:, 3], c="red", marker='o', label='label0')
    plt.scatter(data2[:, 2], data2[:, 3], c="green", marker='*', label='label1')
    # 绘制簇的质心点
    for i in range(length):
        plt.annotate('center', xy=(center[i, 2], center[i, 3]), xytext= \
            (center[i, 2] + 1, center[i, 3] + 1), arrowprops=dict(facecolor='yellow'))
    plt.show()

def draw_k3(data, center, assment):
    # 取前两个维度（萼片长度、萼片宽度），绘制数据分布图
    plt.scatter(fdata[:, 0], fdata[:, 1], c="red", marker='o', label='see')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.legend(loc=2)
    plt.show()
    length = len(center)
    fig = plt.figure
    data1 = data[np.nonzero(assment[:, 0].A == 0)[0]]
    data2 = data[np.nonzero(assment[:, 0].A == 1)[0]]
    data3 = data[np.nonzero(assment[:, 0].A == 2)[0]]
    # 选取前两个维度绘制原始数据的散点图
    plt.scatter(data1[:, 0], data1[:, 1], c="red", marker='o', label='label0')
    plt.scatter(data2[:, 0], data2[:, 1], c="green", marker='*', label='label1')
    plt.scatter(data3[:, 0], data3[:, 1], c="blue", marker='+', label='label2')
    # 绘制簇的质心点
    for i in range(length):
        plt.annotate('center', xy=(center[i, 0], center[i, 1]), xytext= \
            (center[i, 0] + 1, center[i, 1] + 1), arrowprops=dict(facecolor='yellow'))
        #  plt.annotate('center',xy=(center[i,0],center[i,1]),xytext=\
        # (center[i,0]+1,center[i,1]+1),arrowprops=dict(facecolor='red'))
    plt.show()

    # 取后两个维度（萼片长度、萼片宽度），绘制数据分布图
    plt.scatter(fdata[:, 2], fdata[:, 3], c="red", marker='o', label='see')
    plt.xlabel('petal length')
    plt.ylabel('petal width')
    plt.legend(loc=2)
    plt.show()
    # 选取后两个维度绘制原始数据的散点图
    plt.scatter(data1[:, 2], data1[:, 3], c="red", marker='o', label='label0')
    plt.scatter(data2[:, 2], data2[:, 3], c="green", marker='*', label='label1')
    plt.scatter(data3[:, 2], data3[:, 3], c="blue", marker='+', label='label2')
    # 绘制簇的质心点
    for i in range(length):
        plt.annotate('center', xy=(center[i, 2], center[i, 3]), xytext= \
            (center[i, 2] + 1, center[i, 3] + 1), arrowprops=dict(facecolor='yellow'))
    plt.show()

def draw_k4(data, center, assment):
    # 取前两个维度（萼片长度、萼片宽度），绘制数据分布图
    plt.scatter(fdata[:, 0], fdata[:, 1], c="red", marker='o', label='see')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.legend(loc=2)
    plt.show()
    length = len(center)
    fig = plt.figure
    data1 = data[np.nonzero(assment[:, 0].A == 0)[0]]
    data2 = data[np.nonzero(assment[:, 0].A == 1)[0]]
    data3 = data[np.nonzero(assment[:, 0].A == 2)[0]]
    data4 = data[np.nonzero(assment[:, 0].A == 3)[0]]
    # 选取前两个维度绘制原始数据的散点图
    plt.scatter(data1[:, 0], data1[:, 1], c="red", marker='o', label='label0')
    plt.scatter(data2[:, 0], data2[:, 1], c="green", marker='*', label='label1')
    plt.scatter(data3[:, 0], data3[:, 1], c="blue", marker='+', label='label2')
    plt.scatter(data4[:, 0], data4[:, 1], c="black", marker='D', label='label3')
    # 绘制簇的质心点
    for i in range(length):
        plt.annotate('center', xy=(center[i, 0], center[i, 1]), xytext= \
            (center[i, 0] + 1, center[i, 1] + 1), arrowprops=dict(facecolor='yellow'))
        #  plt.annotate('center',xy=(center[i,0],center[i,1]),xytext=\
        # (center[i,0]+1,center[i,1]+1),arrowprops=dict(facecolor='red'))
    plt.show()

    # 取后两个维度（萼片长度、萼片宽度），绘制数据分布图
    plt.scatter(fdata[:, 2], fdata[:, 3], c="red", marker='o', label='see')
    plt.xlabel('petal length')
    plt.ylabel('petal width')
    plt.legend(loc=2)
    plt.show()
    # 选取后两个维度绘制原始数据的散点图
    plt.scatter(data1[:, 2], data1[:, 3], c="red", marker='o', label='label0')
    plt.scatter(data2[:, 2], data2[:, 3], c="green", marker='*', label='label1')
    plt.scatter(data3[:, 2], data3[:, 3], c="blue", marker='+', label='label2')
    plt.scatter(data4[:, 2], data4[:, 3], c="black", marker='D', label='label3')
    # 绘制簇的质心点
    for i in range(length):
        plt.annotate('center', xy=(center[i, 2], center[i, 3]), xytext= \
            (center[i, 2] + 1, center[i, 3] + 1), arrowprops=dict(facecolor='yellow'))
    plt.show()

if __name__ == '__main__':
    fdata = LoadData()
    k = 4
    centroids, clusterAssment = KMeans(fdata, k)

    # if k==2:
    #     draw_k2(fdata, centroids, clusterAssment)
    # elif k==3:
    #     draw_k3(fdata, centroids, clusterAssment)
    # elif k==4:
    #     draw_k4(fdata, centroids, clusterAssment)
    #
    # kx = []
    # sse = []
    # for i in range(1,k+1):
    #     kx.append(i)
    #     centroids, clusterAssment, sseNew = kMeansSSE(fdata, i)
    #     sse.append(sseNew[0,-1])
    # # 绘制损失值
    # plt.figure(figsize=(8, 6))
    # plt.plot(kx, sse)  # 绘制不同k下的效果
    # plt.xlabel('k')
    # plt.ylabel("loss")
    # plt.show()





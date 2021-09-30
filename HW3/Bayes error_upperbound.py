from sklearn.model_selection import train_test_split
import numpy as np
import math
import matplotlib.pyplot as plt
def classifyClass(feature,target):
    class1=[]
    class2=[]
    for i in range(len(target)):
        if target[i]==1:
            class1.append(feature[i])
        else:
            class2.append(feature[i])
    return class1,class2

def data_process():
    with open("/user_data/PR/HW3/dataset.txt","r") as f:
        data = f.read().split('\n')
        data.pop(-1) # 刪除list最後一項空白
        feature = []
        target = []
        for i in data:
            
            i = i.split(",")    
            float_i = []
            for item in i[0:-1]: #資料沒做好 切除最後一項空白
                float_item = float(item)
                float_i.append(float_item)
            feature.append(float_i[1:])
            target.append(int(i[0]))

        feature = np.array(feature)
        target = np.array(target)
        # print(data)
        class1,class2=classifyClass(feature,target)
        return class1,class2

def calculate_bayes_error(classA,classB):
    
    # 參照講義的Bhattacharyya bound公式計算出error
    mean_distance = (np.transpose(np.mean(classB,axis=0)-np.mean(classA,axis=0)).dot(np.linalg.inv((np.cov(np.transpose(classA))+np.cov(np.transpose(classB)))/2)).dot(np.mean(classB,axis = 0)-np.mean(classA)))/8
    cov_distance = np.log( np.linalg.det((np.cov(np.transpose(classA))+np.cov(np.transpose(classB)))/2) / np.sqrt(np.linalg.det(np.cov(np.transpose(classA))) * np.linalg.det(np.cov(np.transpose(classB)))) ) /2
    # print(mean_distance,cov_distance)
    total_dis = mean_distance + cov_distance
    p_a = len(classA)/(len(classA)+len(classB))
    p_b = len(classB)/(len(classA)+len(classB))
    error = np.sqrt(p_a * p_b) * math.exp(-total_dis)
    return error



if __name__ == "__main__":
    class1,class2=data_process()
    print("Bhattacharyya bound(m1,m2)=")
    print(calculate_bayes_error(class1,class2))
    

        
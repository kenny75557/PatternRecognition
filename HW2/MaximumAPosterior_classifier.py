from sklearn.model_selection import train_test_split
import numpy as np
import math

#使用了scikit-learn中套件來切割資料
def data_process(test_size,random_state):
    with open("/user_data/PR/HW2/dataset.txt","r") as f:
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
        x_train, x_test, y_train, y_test = train_test_split(feature, target, test_size=test_size, random_state = random_state)   
        return x_train, x_test, y_train, y_test



#將資料依照target切開
def splitClass(feature,target):
    class1 = []
    class2 = []
    for i in range(len(target)):
        if target[i] == 1:
            class1.append(feature[i])
        elif target[i] ==2:
            class2.append(feature[i])
       
    return class1,class2
# 在給定的特徵時，計算每個label可以得到的機率
def classCalculateProbability(x, train_class,p):
    var = x - np.mean(train_class,axis = 0)
    trans_var = np.transpose(var)
    inv_cov = np.linalg.inv(np.cov(np.transpose(train_class)))
    cov_x = np.cov(np.transpose(train_class))

    return -np.log(p) + var.dot(inv_cov).dot(trans_var)/2 + np.log(np.linalg.det(cov_x))/2

    
def classifier(x, train_class1, train_class2):
    p1 = len(train_class1)/len(train_class1 + train_class2 )
    p2 = len(train_class2)/len(train_class1 + train_class2 )
    

    predict_list = []
    for i in range(len(x)):
        class1_rate = classCalculateProbability(x[i],train_class1,p1)
        class2_rate = classCalculateProbability(x[i],train_class2,p2)
        #貝式分類計算兩class機率
        
        if min(class1_rate, class2_rate) == class1_rate:
            predict_list.append(1)
        elif min(class1_rate, class2_rate) == class2_rate:
            predict_list.append(2)
        
    return predict_list
#將兩個label list(predict&answer)計算百分比
def evaluation(predict,real):
    count = 0
    for i in range(len(predict)):
        if predict[i] == real[i]:
            count += 1
    return count/len(predict) * 100

if __name__ == "__main__":
    test_size = 0.5
     # test dataset 的比例
    random_state = 1
    #隨機劃分的亂數種子
    x_train, x_test, y_train, y_test = data_process(test_size,random_state)
    train_class1, train_class2 = splitClass(x_train,y_train)
    
    predict = classifier(x_test, train_class1, train_class2)
    print("Accuracy:",evaluation(predict,y_test),"%")

    
        
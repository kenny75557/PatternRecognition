from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy as np
import math



def data_process(test_size,random_state):
    with open("/user_data/PR/HW4/dataset.txt","r") as f:
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
        train_data, test_data, train_lab,  test_lab = train_test_split(feature, target, test_size=test_size, random_state = random_state)   #
        return train_data, test_data, train_lab,  test_lab

def model_config():
    model = Sequential()
    model.add(Dense(64, input_dim=50, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam',  metrics=['accuracy'])
    return model


if __name__ == "__main__":
    x_train = np.random.random((1000, 50))
    y_train = np.random.randint(2, size=(1000, 1))
    x_test = np.random.random((100, 50))
    y_test = np.random.randint(2, size=(100, 1))
    test_size = 0.5
     # test dataset 的比例
    random_state = 1
    #隨機劃分的亂數種子
    train_data, test_data, train_lab,  test_lab = data_process(test_size,random_state)
    print(test_data[30],test_lab[30])
    print(len(train_data),len(test_data))
    md=model_config()
    md.fit(train_data, train_lab, epochs=30, batch_size=128)
    score = md.evaluate(test_data, test_lab,batch_size=128)
    print(f'Loss/Accuracy: {score}')



        
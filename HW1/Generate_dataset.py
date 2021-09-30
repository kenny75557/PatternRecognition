import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

#刻出題目要求的covariance matrix,且計算出eigenvector跟eigenvalue
def covar_make(p):
    sigma = np.eye(50)#對角矩陣

    for i in range(50):
        count = 1
        for j in range(50):
            if j > i:
                sigma[i][j] = p ** count#次方
                count += 1
    
    sigma = sigma + sigma.T - np.diag([1] * 50)#上三角+下三角-中間算到兩次(基本矩陣)
    eva, eve = LA.eig(sigma)
    eva = np.diag(eva)#取出對角eigenvector
    
    return eva, eve

eva_m1, eve_m1 = covar_make(0.9)
eva_m2, eve_m2 = covar_make(0.7)
# 將高斯亂數產生器產生的資料做 whitening
def whitening():
    x = np.random.normal(
        0, 1, size=(10000, 50)
    ).T  # 題目是50 x 10000 (50 維資料有 10000 筆)，故轉置

    cov_x = np.cov(x)
    va, ve = LA.eig(cov_x)

    va = np.diag(va)

    w = np.dot(ve, LA.inv(va ** 0.5))
    distribut = np.dot(w.T, x)
    return distribut
temp = whitening()
x = np.dot(np.dot(eve_m1, eva_m1 ** 0.5), temp)#inverse whitening
y = np.dot(np.dot(eve_m2, eva_m2 ** 0.5), temp)
y += 0.5

##check
#plt.plot(x[0], x[1], 'x')
#plt.axis('equal')
#plt.show()
# x_mean = np.mean(x)
# y_mean = np.mean(y)
# print(f'M1: {x_mean}')
# print(f'M2: {y_mean}')

x_cov = np.cov(x)
y_cov = np.cov(y)
nl = '\n'
print(f"Cov1: {nl}{x_cov}")
print(f"Cov2: {nl}{y_cov}")
x=np.transpose(x)
f =open("dataset.txt",'w')
for data in x:
    f.write('0 ')
    for dim in data:
        f.write(str(dim)+' ')
    f.write('\n')


y=np.transpose(y)
#f =open("M2.txt",'w')
for data in y:
    f.write('1 ')
    for dim in data:
        f.write(str(dim)+' ')
    f.write('\n')
f.close
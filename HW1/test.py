import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt


# 提取出 eigenvalue, eigenvector
def eigen_extractor(cov):
    eva, eve = LA.eig(cov)
    eva = np.diag(eva)
    return eva, eve


# 將高斯亂數產生器產生的資料做 whitening
def whitening(eva, eve, data):
    w = np.dot(eve, LA.inv(eva ** 0.5))
    identity_data = np.dot(w.T, data)
    return identity_data


# 產生出老師給的 covariance matrix
def cov_generator(p):
    sigma = np.eye(50)
    for i in range(50):
        count = 1
        for j in range(50):
            if j > i:
                sigma[i][j] = p ** count
                count += 1
    sigma = sigma + sigma.T - np.diag([1] * 50)
    return sigma


# 做反向的 whitening
def inverse_whitening(eva, eve, data):
    new_data = np.dot(np.dot(eve, eva ** 0.5), data)
    return new_data


# 產生出高斯分布亂數並轉置（矩陣 10000 x 50，代表有 10000 筆長度為 50 的資料，但投影片的公式是假設長度 50 的資料有 10000 筆，也就是 50 x 10000，所以這邊要轉置）
random_num = np.random.normal(0, 1, size=(10000, 50)).T

# 提取出亂數的 covariance matrix
random_cov = np.cov(random_num)

# 提取出亂數的 eigenvalue, eigenvector
r_eva, r_eve = eigen_extractor(random_cov)

# 對亂數做 whitening
identity_data = whitening(r_eva, r_eve, random_num)

# 產生老師規定的 covariance matrix
cov1 = cov_generator(0.9)
cov2 = cov_generator(0.7)


# 提取老師給的 cov 的 eigenvalue, eigenvector
eva1, eve1 = eigen_extractor(cov1)
eva2, eve2 = eigen_extractor(cov2)

# 用反向的 whitening 製造出新資料
x = inverse_whitening(eva1, eve1, identity_data)
y = inverse_whitening(eva2, eve2, identity_data)

# 因為平均值為 0.5 所以 y 的所有 elements 都要加 0.5
y += 0.5

# 驗證其 mean 跟 covariance matrix
x_mean = np.mean(x)
y_mean = np.mean(y)
print(f'M1: {x_mean}')
print(f'M2: {y_mean}')

x_cov = np.cov(x)
y_cov = np.cov(y)
nl = '\n'
print(f"Cov1: {nl}{x_cov}")
print(f"Cov2: {nl}{y_cov}")
x=np.transpose(x)
f =open("dataset.txt",'w')
for data in x:
    f.write('0,')
    for dim in data:
        f.write(str(dim)+',')
    f.write('\n')


y=np.transpose(y)
#f =open("M2.txt",'w')
for data in y:
    f.write('1,')
    for dim in data:
        f.write(str(dim)+',')
    f.write('\n')
f.close
# plt.plot(y[0], y[1], 'y')
# plt.axis('equal')
# plt.show()
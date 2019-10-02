# --*-- coding:gb18030 --*--

import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.mplot3d import Axes3D
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import os
import time

print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
# ��ȡ�ļ�
os.chdir('D:/data//train_set_process_by_column_add_xy_distance_level_relative_degree/')
#os.chdir('G:/other/temp/')
file_chdir = os.getcwd()
filecsv_list = []
for root,dirs,files in os.walk(file_chdir):
    for file in files:
        if os.path.splitext(file)[1] == '.csv':
            filecsv_list.append(file)
data = pd.DataFrame()
test_arr=[]
for csv in filecsv_list:
    temp = pd.read_csv(csv,header = 0,sep=',',encoding='gb18030')
    # test_arr.append(temp)
    # filesDatas = np.array(temp, dtype=np.float32)
    # print(test_arr)
    if temp.size == 0:
        print(csv)
        os.remove(csv)
        continue
    temp = temp.sample(frac=1.0)     # ȫ������
    data = data.append(temp.head(2))  # ȡ���С����ǰ��2��

#df=pd.read_csv('D:\data/train_set/train_108401.csv')

df = data

data = df.astype(float)
#data = data.drop(columns = 'x_distance')
#data = data.drop(columns='y_distance')
#data = data.drop(columns='y_distance')

# print(data)
data_corr = data
X = data.iloc[:,:-1]
Y = data['RSRP']
pca = PCA(n_components=4)
new_X = pca.fit_transform(X)    # ����ѵ��pcaģ�ͣ����ؽ�ά֮��Ľ��
print("�����ɷ���Ϣռ��",pca.explained_variance_ratio_)
corr_matrix = data_corr.corr(method='pearson')['RSRP']
print(corr_matrix)

#��������
train_set1, test_set2,train_lab1,test_lab2 = sklearn.model_selection.train_test_split(X,Y, test_size=0.2,random_state=1)#���ݼ�����
print(len(train_set1),len(test_set2))  # ���ѵ�������Ի���С
train_set111 = pca.transform(train_set1) # pca��ά����
train_set222 = pca.transform(test_set2)    # pca��ά
# train_set111 = train_set1
# train_set222 = test_set2
train_set11 = MinMaxScaler(feature_range=(0, 1)).fit_transform(train_set111)#���ݹ�һ��
test_set22 = MinMaxScaler(feature_range=(0, 1)).fit_transform(train_set222)#���ݹ�һ��
#print(train_set11,test_set22)

#data���ݴ������
# fc = KernelRidge()
# fc = sklearn.ensemble.GradientBoostingRegressor()
sum = 0
min = 500
for i in range(0,20):  # 20==>1
    # ���ɭ�ֻع�
    fc = RandomForestRegressor()
    # �ݶ������ع�
    #fc = sklearn.ensemble.GradientBoostingRegressor()
    fc.fit(train_set11,train_lab1)
    pro = fc.predict(test_set22)

    # fc = lgb.LGBMRegressor()
    # fc.fit(train_set11,train_lab1)
    # pro = fc.predict(test_set22)


    mse = sklearn.metrics.mean_squared_error(test_lab2, pro)
    sum = sum + np.sqrt(mse)
    print(i)
    if np.sqrt(mse)<min:
        min = np.sqrt(mse)
print("RMSE: %.4f" % (sum/20))
print("min: %.4f" % min)
chazhi = 0
for j in range(len(pro)):
    t1 = test_lab2.iat[j]
    t2 = pro[j]
    chazhi = chazhi + abs(t1-t2)
print("ƽ����ֵ(RSRP): %.4f" % (chazhi/len(pro)))
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
#3D��ͼ��ʼ
fig = plt.figure()
ax3d=Axes3D(fig)
# for i in range(len(test_set2['lon'])):
#     ax3d.plot(test_set2['lon'][i],test_set2['lat'][i],test_lab2[i], color='b')
ax3d.scatter(test_set2['x_distance'],test_set2['y_distance'],test_lab2,color = 'blue',label = 'real',alpha=0.5)
ax3d.scatter(test_set2['x_distance'],test_set2['y_distance'],pro,color = 'red',label = 'predict',alpha=0.5)
for i in range(len(test_set2['x_distance'])):
    x1=x2=test_set2['x_distance'].iat[i]
    y1=y2=test_set2['y_distance'].iat[i]
    z1=test_lab2.iat[i]
    z2=pro[i]
    # print(z1-z2)
    ax3d.plot([x1,x2],[y1,y2],[z1,z2],color='k',alpha=0.5)
ax3d.set_xlabel('x_distance')
ax3d.set_ylabel('y_distance')
ax3d.set_zlabel('RSRP')
ax3d.legend(loc = 'best')
plt.show()
#3D��ͼ���



#������Ҫ������ʼ

# df=pd.read_csv('D:\data/train_set/train_108401.csv')
#
# data = df.astype(float)
# data = df.drop(columns='Cell Index')
# data = data.astype(float)
# data_corr = data
# X = data.iloc[:,:-1]
# Y = data['RSRP']

# print(X,Y)
fc = RandomForestRegressor()
#fc = sklearn.ensemble.GradientBoostingRegressor()
#fc = lgb.LGBMRegressor()
fc.fit(X,Y)

feature_importances = fc.feature_importances_
lieming = X.columns
print(feature_importances)
plt.title("feature_importances")
plt.bar(lieming,feature_importances,color ='lightblue',align='center')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
#������Ҫ���������
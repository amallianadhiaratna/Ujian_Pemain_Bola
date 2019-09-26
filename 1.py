import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


df=pd.read_csv('data.csv')
# print(df.columns)

df=df[['Name','Age','Overall','Potential']]
print(df.head())

rec = df[df['Age']<=25]
rec = rec[rec['Overall']>=80] 
rec = rec[rec['Potential']>=80]
rec['status'] = 1
# print(rec.head())
norec= df.drop(rec.index)
norec['status'] = 0
# print(norec.head())

train = pd.concat([rec, norec], axis=0).drop(columns="Name")
# print(len(train))


def k():
    k = round(len(train) ** .5)
    if k % 2 == 0:
        return k + 1
    else:
        return k 
              
print(cross_val_score(
    KNeighborsClassifier(n_neighbors=k()), train[['Age', 'Overall', 'Potential']], train['status'],
    cv = 4))
print(cross_val_score(
    RandomForestClassifier(n_estimators=100), train[['Age', 'Overall', 'Potential']], train['status'],
    cv = 4))
print(cross_val_score(
    DecisionTreeClassifier(), train[['Age', 'Overall', 'Potential']], train['status'],
    cv = 4))

acc_knn = np.mean(np.array([0.80669962, 0.99972543, 0.99697968, 0.99423235, 0.99093407]))
print(f'Akurasi KNNeighbours : {acc_knn*100} %')
acc_ranfor = np.mean(np.array([0.7188358,  1.,         1.,         1.,         0.99725275]))
print(f'Akurasi Random Forest Classifier: {acc_ranfor*100} %')
acc_dectree = np.mean(np.array([0.7188358, 1.,        1.,        1.,        1.       ]))
print(f'Akurasi Decision Tree Classifier : {acc_dectree*100} %')

knn = KNeighborsClassifier(n_neighbors=k())
knn.fit(train[['Age', 'Overall', 'Potential']], train['status'])

df_indo = pd.DataFrame(np.array([           
    ['Andik Vermansyah', 'Madura United FC', 27, 87, 90], 
    ['Awan Setho Raharjo', 'Bhayangkara FC', 22, 75, 83],
    ['Bambang Pamungkas', 'Persija Jakarta', 38, 85, 75],
    ['Cristian Gonzales', 'PSS Sleman', 43, 90, 85],
    ['Egy Maulana Vikri', 'Lechia Gdańsk', 18, 88, 90],
    ['Evan Dimas', 'Barito Putera', 24, 85, 87],
    ['Febri Hariyadi', 'Persib Bandung', 23, 77, 80],
    ['Hansamu Yama Pranata', 'Persebaya Surabaya', 24, 82, 85],
    ['Septian David Maulana', 'PSIS Semarang', 22, 83, 80],
    ['Stefano Lilipaly', 'Bali United', 29, 88, 86]]),
    columns=['Name', 'Club', 'Age', 'Overall', 'Potential']
)
df_indo['Nationality'] = 'Indonesia'
print(df_indo)

target = ['no', 'yes']

df_indo['Target'] = knn.predict(df_indo[['Age', 'Overall', 'Potential']])
df_indo['Target'] = df_indo['Target'].apply(lambda i: target[i])
print(df_indo)

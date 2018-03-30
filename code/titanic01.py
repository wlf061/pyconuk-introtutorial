import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("data/train.csv")
print(df.head(10))
df =df.drop(['Name','Ticket','Cabin'], axis=1)
print(df.info()) ## 通过info查看到有些字段为N/A， 通过dropna 删除空的字段
df = df.dropna()

###Sciket-learn的输入只支持 数字，需要对Sex和Embarked 进行mapping
df['Gender'] = df['Sex'].map({'female':0,'male':1}).astype(int)
df['Port'] = df['Embarked'].map({'C':1, 'S':2, 'Q':3}).astype(int)
df =df.drop(['Sex','Embarked'], axis=1)

###将Survived 列移动到最左边
cols = df.columns.tolist()

## 或者 cols.insert(0, cols.pop(cols.index('Survived')))
cols = [cols[1]] + cols[0:1] + cols[2:]
df = df[cols]  ##或者 df.reindex(columns=cols)


#获取训练数据
train_data = df.values

#使用随机森林
model = RandomForestClassifier(n_estimators = 100)

model = model.fit(train_data[0:,2:], train_data[0:,0])

##处理测试数据

df_test = pd.read_csv('data/test.csv')
df_test = df_test.drop(['Name', 'Ticket', 'Cabin'], axis=1)

df_test = df_test.dropna()

df_test['Gender'] = df_test['Sex'].map({'female': 0, 'male':1})
df_test['Port'] = df_test['Embarked'].map({'C':1, 'S':2, 'Q':3})

df_test = df_test.drop(['Sex', 'Embarked'], axis=1)

test_data = df_test.values
output = model.predict(test_data[:,1:])

##处理数据集
result = np.c_[test_data[:,0].astype(int), output.astype(int)]
df_result = pd.DataFrame(result[:,0:2], columns=['PassengerId', 'Survived'])

print(df_result.head(10))

df_result.to_csv('data/titanic_1-0.csv', index=False)



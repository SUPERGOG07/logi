import pandas as pd

train = pd.read_csv("./data/train.csv")
train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
train.Age.fillna(train.Age.mode()[0], inplace=True)
train.dropna(inplace=True)
train = pd.get_dummies(train)
train = train.replace({True: 1, False: 0})
train.to_csv("./pre_data/train.csv", index=False)

#合并
gender_submission = pd.read_csv("./data/gender_submission.csv")
test = pd.read_csv("./data/test.csv")
test = test.merge(gender_submission, on='PassengerId', how='left')
test['Survived'] = test['Survived'].fillna(0).astype(int)

# 重新排列列顺序，将Survived列放到第一列位置
columns = list(test.columns)
columns.remove('Survived')
columns.insert(0, 'Survived')
test = test[columns]

test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
test.Age.fillna(test.Age.mode()[0], inplace=True)
test.dropna(inplace=True)
test = pd.get_dummies(test)
test = test.replace({True: 1, False: 0})
test.to_csv("./pre_data/test.csv", index=False)

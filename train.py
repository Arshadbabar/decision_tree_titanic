import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

df=pd.read_csv('data/TitanicData.csv')
# print(df.head())
df.isnull().sum()

X=df.drop(columns='Survived')
y=df['Survived']

# print(X.shape)
# print(y.shape)

X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,random_state=42
)
X_test.shape
X_train.shape

model=DecisionTreeClassifier()
model.fit(X_train,y_train)

y_pred=model.predict(X_test)
print('Actual:',y_test[:5].values)
print('Prediction',y_pred[:5])

ac=accuracy_score(y_test,y_pred)
pre=precision_score(y_test,y_pred)
rec=recall_score(y_test,y_pred)
f1=f1_score(y_test,y_pred)

print('Accuracy:',ac)
print('Precision:',pre)
print('Recall',rec)
print('F1',f1)

# Hyperparameter Optimization for performance increase

param_grid = [{
    'criterion': ['entropy', 'gini'],
    'max_depth': [3, 4, 5, None],
    'min_samples_split': [10, 11],
    'min_samples_leaf': [2,3]
}]

dt_grid=GridSearchCV(
    DecisionTreeClassifier(),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1

)

dt_grid.fit(X_train, y_train)
best_grid = dt_grid.best_estimator_
dt_pred = best_grid.predict(X_test)
print('Best model selection',best_grid)

print('Actual value:',y_test[:5].values)
print('Prediction using GridSearch:',dt_pred[:5])

dt_accuracy=accuracy_score(y_test,dt_pred)
dt_precision=precision_score(y_test,dt_pred)
dt_recall=recall_score(y_test,y_pred)
dt_f1=f1_score(y_test,y_pred)

# Performance measure after Hyperparametre Optimization

print('Accuracy',dt_accuracy)
print('Precision',dt_precision)
print('Recall',dt_recall)
print('f1',dt_f1)

# Final comparision Report
df=pd.DataFrame({
    'Model':['Default_Model','GridSearchCV'],
    'Accuracy':[ac,dt_accuracy],
    'Precision':[pre,dt_precision],
    'Recall':[rec,dt_recall],
    'F1':[f1,dt_f1]

}
)
print(df)


# 1 - import packges : 
import numpy as np
import pandas as pd
import sklearn as sk

# 2 - reed the file : 
df = pd.read_csv('C:\\Users\\moham\\Desktop\\VScode\\Diabets\\diabetes.csv')
df = pd.DataFrame(df)

# 3 - get some informaition about the data :
# print(df.head(5))
# print(df.shape)
# print(df.columns)
# print(df.dtypes)
# print(df.info())
# print(df.isnull().sum())
# print(df.describe())
# print(df['Outcome'].value_counts())



# 3 - split the data :
X = df.drop(columns='Outcome')
Y = df['Outcome']

# 4- train the data :
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.20 , random_state=0)
# print('x_train size: {}, X_test size: {}'.format(x_train.shape, x_test.shape))

# 5 - Data Preprocessing :
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

# 6 - import the packges for models : 
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# 7 - choose the best model : 
def find_best_model(X, y):
    models = {
        'logistic_regression': {
            'model': LogisticRegression(solver='lbfgs', multi_class='auto'),
            'parameters': {
                'C': [1,5,10]
               }
        },
        
        'decision_tree': {
            'model': DecisionTreeClassifier(splitter='best'),
            'parameters': {
                'criterion': ['gini', 'entropy'],
                'max_depth': [5,10]
            }
        },
        
        'random_forest': {
            'model': RandomForestClassifier(criterion='gini'),
            'parameters': {
                'n_estimators': [10,15,20,50,100,200]
            }
        },
        
        'svm': {
            'model': SVC(gamma='auto'),
            'parameters': {
                'C': [1,10,20],
                'kernel': ['rbf','linear']
            }
        }

    }
    
    scores = [] 
    cv_shuffle = ShuffleSplit(n_splits=5, test_size=0.20, random_state=0)
        
    for model_name, model_params in models.items():
        gs = GridSearchCV(model_params['model'], model_params['parameters'], cv = cv_shuffle, return_train_score=False)
        gs.fit(X, y)
        scores.append({
            'model': model_name,
            'best_parameters': gs.best_params_,
            'score': gs.best_score_
        })
        
    return pd.DataFrame(scores, columns=['model','best_parameters','score'])

# print(find_best_model(X_train, y_train))

# 8 - fit the model :
svm_model = SVC(C=20, kernel='rbf', gamma='auto')
svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)

#9 - Evaluate classification model performance :
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)



# 10 - save the model :
import pickle
pickle.dump(svm_model,open('diabetes.pkl','wb'))
model = pickle.load(open( 'diabetes.pkl', 'rb' ))

# now we can try the model : 
prediction_data = pd.DataFrame(data=np.array([2,197,50,34,90,20,0.243,65]).reshape(1,8))
prediction = model.predict(prediction_data)
if prediction:
  print('Positive')
else:
  print("Negative")
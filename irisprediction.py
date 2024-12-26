from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data=load_iris()
df=pd.DataFrame(data.data,columns=data.feature_names)
df['target']=data.target

X=df[['petal length (cm)', 'petal width (cm)']]
y=df['target']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)


models = {
    'Logistic Regression': LogisticRegression(max_iter=200, random_state=42),
    'K-Nearest Neighbors (KNN)': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42)
}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f'{model_name} Accuracy: {accuracy * 100:.2f}%')

    
    print(f'{model_name} Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))

    
    print(f'{model_name} Precision, Recall, F1-Score:')
    print(classification_report(y_test, y_pred))

    print('-' * 50) 

plt.scatter(df['petal length (cm)'],df['petal width (cm)'],c=df['target'],cmap='viridis')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('Petal Length vs Width')
plt.show()





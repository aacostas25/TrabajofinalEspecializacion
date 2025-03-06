from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

regressors={
    'DecisionTreeClassifier':(DecisionTreeClassifier(),
                   {
                       'reg__max_depth':[1,2,3,4,5,6,7,8,9,10,11,15,18,20,21]
                   }),
}
scaler={

    'None':None
}

X = basemedica.iloc[:, :-1]
y = basemedica['Cath']

X_encoded = pd.get_dummies(X, drop_first=True)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)

best_model=None
best_accuracy=-float('inf')
for scaler_name,scaler_value in scaler.items():
    for regressor_name,(regressor,param_grid) in regressors.items():
        print(f'Probando {regressor_name} con {scaler_name}')
        steps=[('reg',regressor)]
        if scaler_value is not None:
            steps.insert(0,('scaler',scaler_value))
        model=Pipeline(steps)
        grid_search=GridSearchCV(model,param_grid,cv=5,n_jobs=-1,scoring='balanced_accuracy',verbose=0)
        grid_search.fit(X_train,y_train)
        y_pred = grid_search.predict(X_test)
        accuracy= metrics.accuracy_score(y_test,y_pred)
        print(f'AUC: {accuracy}')
        print(f'Mejores hiperparametros: {grid_search.best_params_}')
        if accuracy>best_accuracy:
            best_accuracy=accuracy
            best_model=(regressor_name,scaler_name,grid_search.best_estimator_)
print(f'Mejor modelo: {best_model}')
print(f'Mejor accuracy: {best_accuracy}')

# Suponiendo que 'best_model' es el modelo entrenado con max_depth=3
decision_tree = best_model[2].named_steps['reg']  # Extraer el árbol del pipeline

plt.figure(figsize=(15, 8))  # Ajustar el tamaño del gráfico
plot_tree(decision_tree,
          feature_names=X_encoded.columns,  # Nombres de las variables
          class_names=['Normal', 'Cad'],  # Etiquetas de las clases
          filled=True,
          rounded=True,
          fontsize=10)

plt.savefig("decision_tree.png", dpi=300, bbox_inches="tight")  # Guardar imagen
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Extraer la importancia de las características
importances = decision_tree.feature_importances_
feature_names = X_encoded.columns

# Crear un DataFrame para ordenar los valores
importance_df = pd.DataFrame({"Variable": feature_names, "Importancia": importances})
importance_df = importance_df.sort_values(by="Importancia", ascending=False)

# Graficar usando Seaborn
plt.figure(figsize=(12, 6))
sns.barplot(x=importance_df["Importancia"], y=importance_df["Variable"], palette="viridis")

plt.xlabel("Importancia")
plt.ylabel("Variable")
plt.title("Importancia de las variables en el Árbol de Decisión")

# Guardar la imagen
plt.savefig("importancia_variables.png", dpi=300, bbox_inches="tight")
plt.show()

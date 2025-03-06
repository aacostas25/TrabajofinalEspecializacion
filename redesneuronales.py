# 1️ Instalación automática de wandb si no está instalado
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

try:
    import wandb
except ImportError:
    !pip install wandb
    import wandb

#  2️Iniciar sesión con API Key (sin intervención manual)
wandb.login(key=YOUR_API_KEY)  #  Reemplaza con tu clave real

# 3️Cargar dependencias necesarias
from wandb.integration.keras import WandbCallback
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# 4️ Cargar y normalizar datos MNIST
import pandas as pd
basemedica=pd.read_excel("/content/BD Taller final Minería de datos.xlsx")

X = basemedica.iloc[:, :-1]
y = basemedica['Cath']

X_encoded = pd.get_dummies(X, drop_first=True)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)



# 5️ Definir configuración del Sweep con q_normal en batch_size
sweep_config = {
    "method": "random",  # Métodos: "grid", "random", "bayes"
    "metric": {"name": "val_loss", "goal": "minimize"},  # Minimizar val_loss
    "parameters": {
        "epochs": {"distribution": "int_uniform", "min": 4, "max": 12},
        "batch_size": {"distribution": "int_uniform", "min": 8, "max": 64},
        "learning_rate": {"distribution": "log_uniform_values", "min": 0.0001, "max": 0.1},
        "num_units": {"distribution": "int_uniform", "min": 4, "max": 64},  #  Media, sigma paso
        "depth": {"distribution": "int_uniform", "min": 1, "max": 8},  #  Número de capas ocultas aleatorio
        "activation": {"values": ["relu", "tanh", "sigmoid"]},  #  Valores discretos
        "optimizer": {"values": ["adam", "sgd", "rmsprop"]}  # Valores discretos
    }
}

project_name= "prueba_taller15"

# 6️ Crear el Sweep en wandb sin logs innecesarios
sweep_id = wandb.sweep(sweep_config, project=project_name)

# 7️ Definir la función de entrenamiento
def train():
    wandb.init(settings=wandb.Settings(_disable_stats=True, silent=True))  #  Desactivar logs de wandb
    config = wandb.config  # Obtener los hiperparámetros del Sweep

    # Construcción del modelo sin Dropout
    model = keras.Sequential()
    model.add(keras.Input(shape=(x_train.shape[1],)))
    for _ in range(config.depth):
        model.add(layers.Dense(config.num_units, activation=config.activation))  #  Activación configurable
    model.add(layers.Dense(2, activation="softmax"))

    optimizer = {
        "adam": keras.optimizers.Adam(learning_rate=config.learning_rate),
        "sgd": keras.optimizers.SGD(learning_rate=config.learning_rate),
        "rmsprop": keras.optimizers.RMSprop(learning_rate=config.learning_rate)
    }[config.optimizer]

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Evitar el guardado del modelo en WandB
    class CustomWandbCallback(WandbCallback):
        def _save_model_as_artifact(self, *args, **kwargs):
            pass  #  Desactiva el guardado automático del modelo

    # Entrenar el modelo sin verbose
    history = model.fit(
        x_train, y_train,
        epochs=config.epochs,
        batch_size=config.batch_size,
        validation_data=(x_test, y_test),
        verbose=1,  #  Desactiva verbose de Keras
        callbacks=[CustomWandbCallback(log_model=False, save_graph=False, save_code=False)]
    )

    # Evaluar el modelo sin verbose
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)  #  Desactiva verbose

    # Registrar métricas adicionales
    wandb.log({
        "test_accuracy": test_acc,
        "test_loss": test_loss,
        "epochs": config.epochs,
        "batch_size": config.batch_size,  # Ahora batch_size se genera con q_normal
        "learning_rate": config.learning_rate,
        "num_units": config.num_units,
        "depth": config.depth,
        "activation": config.activation,
        "optimizer": config.optimizer
    })
    wandb.finish()  # Finalizar experimento
# 8️ Ejecutar el Sweep sin logs innecesarios
wandb.agent(sweep_id, function=train, count=5)  #  Desactivar logs de wandb.agent
print(" Sweeps completados. Revisa los resultados en: https://wandb.ai")

# veamos los modelos que se entrenaron
wandb.init()

api = wandb.Api()

# Obtener la lista de proyectos
projects = api.projects()
print("Proyectos disponibles:", [p.name for p in projects])

# Obtener todas las ejecuciones del proyecto
runs = api.runs(f"{wandb.run.entity}/{project_name}")

# Encontrar la mejor ejecución basada en la menor val_loss
best_run = min(runs, key=lambda run: run.summary.get("val_loss", float("inf")))

# Obtener los mejores hiperparámetros
best_hyperparams = best_run.config
print("Mejores hiperparámetros encontrados:", best_hyperparams)

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Extraer la importancia de las características
importances = decision_tree.feature_importances_
feature_names = X_encoded.columns

# Crear un DataFrame para ordenar los valores
importance_df = pd.DataFrame({"Variable": feature_names, "Importancia": importances})
importance_df = importance_df[importance_df["Importancia"]>0]
importance_df = importance_df.sort_values(by="Importancia", ascending=False)

# Graficar usando Seaborn
plt.figure(figsize=(12, 6))
sns.barplot(x="Importancia", y="Variable", hue="Variable", data=importance_df, palette="viridis", legend=False)

plt.xlabel("Importancia")
plt.ylabel("Variable")
plt.title("Importancia de las variables en el Árbol de Decisión")

# Guardar la imagen
plt.savefig("importancia_variables.png", dpi=300, bbox_inches="tight")
plt.show()

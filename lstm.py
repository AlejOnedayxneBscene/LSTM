import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import random, time
from mealpy import Problem
from mealpy.swarm_based import SSA
from mealpy.utils.space import IntegerVar, FloatVar

# asignamos un valor aleatorio a los pesos de cada neurona
# y nos aseguramos que estos sean siempre los mismo por cada ejecucion
np.random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Carga de datos
excel_path = r"C:\Users\aleja\OneDrive\Desktop\pytorch\Mejores datos de chapinero.xlsx"
df = pd.read_excel(excel_path)

#remplazamos para que las columnas puedan ser leidas por pandas
df.columns = [c.strip().replace(" ", "_").replace("(", "").replace(")", "")
              .replace("°", "").replace("/", "_").replace(".", "") for c in df.columns]

# tratar el dato como tipo fecha
fecha_cols = [c for c in df.columns if "date" in c.lower() or "hora" in c.lower() or "datetime" in c.lower()]
if not fecha_cols:
    raise ValueError("No se encontró columna de fecha")
fecha_col = fecha_cols[0]

df[fecha_col] = pd.to_datetime(df[fecha_col])
df = df.sort_values(fecha_col).reset_index(drop=True)

# Selección de variable objetivo 
original_target = "PM2.5 (µg/m3)"
target_col = original_target.strip().replace(" ", "_").replace("(", "").replace(")", "") \
    .replace("°", "").replace("/", "_").replace(".", "")

columnas_excluir = [fecha_col]
variables = [c for c in df.columns 
             if c not in columnas_excluir 
             and pd.api.types.is_numeric_dtype(df[c])
             and not df[c].isna().all()]

if df[variables].isna().sum().sum() > 0:
    df[variables] = df[variables].interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')

# usamos el 80% de los datos para el entrenamiento
train_split = 0.8
split_idx = int(train_split * len(df))

X_all = df[variables].values 
# convertimos la variable a predecir en un arreglo bidimentsion
y_all = df[target_col].values.reshape(-1, 1)

X_train_raw = X_all[:split_idx] # entrenamiento
y_train_raw = y_all[:split_idx] # prueba

X_test_raw = X_all[split_idx:] # entrenamiento
y_test_raw = y_all[split_idx:] # prueba

# Normalizacion
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
scaler_X.fit(X_train_raw)
scaler_y.fit(y_train_raw)

X_train_scaled = scaler_X.transform(X_train_raw)
X_test_scaled = scaler_X.transform(X_test_raw)
y_train_scaled = scaler_y.transform(y_train_raw)
y_test_scaled = scaler_y.transform(y_test_raw)

# Crear ventanas temporales
def create_windows(X, y, L):
    Xs, ys = [], []
    for i in range(L, len(X)):
        Xs.append(X[i - L:i])  # toma los valores previos a en las variables i el cual es el valor de la ventana del tiemp
        ys.append(y[i])  # toma el valor de i de la variable a predecir
    return np.array(Xs), np.array(ys)

# Modelo LSTM
class LSTMModel(nn.Module):
    #constructor
    def __init__(self, input_dim, hidden_layers, dropout): # numero de caracteruisticas, cantidad de neuronas por capa
        super(LSTMModel, self).__init__()
        # creacion de capas
        self.lstm_layers = nn.ModuleList() # lista que contiene varias capas LSTM
        input_size = input_dim # tamaño de entrada
        for h in hidden_layers: # iteramos sobre cada numero de neuronas
            self.lstm_layers.append(nn.LSTM(input_size, h, batch_first=True)) # creacion del lstm  los datos 
            input_size = h #actualiza el tamaño de entrada a la siguiente capa 
        self.dropout = nn.Dropout(dropout) # apaga un porcentajke de neuroinas
        self.fc1 = nn.Linear(hidden_layers[-1], 32) #  capa densa con 32 neuronas
        self.fc2 = nn.Linear(32, 1) #  toma las 32 señalkes de la capa densa y la convierte en un unico valor
        self.relu = nn.ReLU() #  a los valores negativcoc introduce no linealidad en la red

    def forward(self, x): # secuencias procesadsas, longitud de la secuencia, variables en cada paso
        for lstm in self.lstm_layers: #lista de lstm 
            x, _ = lstm(x)
        x = x[:, -1, :] # toma el; ultimo paso de la secuencia
        x = self.dropout(x) # apaa neuronas al azar
        x = self.relu(self.fc1(x)) # añade ni linealidad
        x = self.fc2(x)  # toma las 32 señalkes de la capa densa y la convierte en un unico valor
        return x # devuelkve la prediccion

sparrow_counter = 0 # cuantas veces se ha evaluado el modelo
call_count = 0  # cuenta total de llamadas a objective_function
sparrow_history = []

def evaluate_model(params, iter_num=None, sparrow_idx=None):
    global sparrow_counter, sparrow_history
    sparrow_counter += 1 # incrementar cada vez que se evalua el modelo 

    L = int(params["lookback"]) # tamaño de la ventana del tiempo
    num_layers = int(max(1, min(params["num_layers"], MAX_LAYERS))) # numero de capas lstm 
    neuron_values = params["num_neurons"][:num_layers] # numero de neuronas por capas
    

    X_train_local, y_train_local = create_windows(X_train_scaled, y_train_scaled, L)  # enviamos los valores de entrenamiento
    X_test_local, y_test_local = create_windows(X_test_scaled, y_test_scaled, L) # enviamos los valores de test
    
    n_features_local = X_train_local.shape[2] #numero de variab;es
    # instancia de la rd lstm, numero de variables, valor de neuronas por centaje de neuronas apagadas
    model = LSTMModel(n_features_local, neuron_values, params["dropout"]).to(device)
    #ajusta el peso de las neuronas
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
    # calcular error cuadratico medio
    loss_fn = nn.MSELoss()

    # convierte los arreglo de numpy a tensores de pytorch para poder hacer calculos en la gpu
    train_dataset = TensorDataset(torch.tensor(X_train_local, dtype=torch.float32),
                                  torch.tensor(y_train_local, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test_local, dtype=torch.float32),
                                 torch.tensor(y_test_local, dtype=torch.float32))
    
    #número de muestras que la red procesa antes de actualizar los pesos
    train_loader = DataLoader(train_dataset, batch_size=max(1, int(params["batch_size"])), shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=max(1, int(params["batch_size"])), shuffle=False)

    best_loss = float("inf") # guarda el mejor error de validacion
    # cada vez que encontramos un modelo con menor error, actusot pesos
    best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    # paciencia
    patience_counter = 0
    # numero de epocas maximo
    max_epochs = 80


    for epoch in range(max_epochs):
        model.train() # entrena el modelo
        # usamos lod datos de entrada
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device) # envia los datos a la pu
            optimizer.zero_grad()
            loss = loss_fn(model(Xb), yb) # se realiza la prediccion y se compara con el val;or real
            loss.backward() #calcula automaticmane el cambio que hace cada neurona
            optimizer.step()

        model.eval() # evaluacion del modelo 
        val_loss = 0
        with torch.no_grad():
            for Xv, yv in test_loader: # datos y valores reales de validacion
                Xv, yv = Xv.to(device), yv.to(device) # envia los datos a la gpu 
                val_loss += loss_fn(model(Xv), yv).item()  #calculamos el error mse
        val_loss /= len(test_loader)


#comparamos el error de validacion actual con el mejor que hemos tenido antes
        if val_loss < best_loss:
            best_loss = val_loss
            # uarda los persos actuales
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            # devuelve la paciencia a 0
            patience_counter = 0
        else:
            # incrimernta a 1 la paciencia
            patience_counter += 1
            # si alcnaza la paciencia antes del numero de epocas se detiene el entrenamiento
        if patience_counter >= params["patience"]:
            break
# guarda los mejores persos durante el; entrenamiento
    model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})
# evaluamos el modelo
    model.eval()
    with torch.no_grad():
        # convertimos a  los datos del test a un tensor y los enviamos a ka py para obtener las predicciones
        X_test_tensor = torch.tensor(X_test_local, dtype=torch.float32).to(device)
        y_pred_scaled = model(X_test_tensor).cpu().numpy().reshape(-1, 1)
# desnormalizacion
    y_pred_real = scaler_y.inverse_transform(y_pred_scaled).reshape(-1)
    y_test_real = scaler_y.inverse_transform(y_test_local).reshape(-1)
# calcula erro cuadratico y coeficiente de determinacion
    mse = mean_squared_error(y_test_real, y_pred_real)
    r2 = r2_score(y_test_real, y_pred_real)

   # Imprime información de iteración y gorrión si está disponible
    iter_info = f"Iteración={iter_num}" if iter_num is not None else ""
    sparrow_info = f"Gorrión={sparrow_idx:02d}" if sparrow_idx is not None else f"Gorrión=??"
    print(f" {iter_info} | {sparrow_info} | Lookback={L} | Capas={num_layers} | Neuronas={neuron_values} | "
          f"Dropout={params['dropout']:.2f} | LR={params['learning_rate']:.5f} | Batch={params['batch_size']} | "
          f"Patience={params['patience']} | R²={r2:.4f} | MSE={mse:.4f}")

#guarda todos los parametro de cada horrion
    sparrow_history.append({
        "Iteracion": iter_num,
        "Gorrion": sparrow_idx,
        "Lookback": L,
        "Num_Capas": num_layers,
        "Neuronas": neuron_values,
        "Dropout": params["dropout"],
        "LR": params["learning_rate"],
        "Batch": params["batch_size"],
        "R2": r2,
        "MSE": mse
    })


    return mse, model

# Función objetivo SSA
MAX_LAYERS = 5
def objective_function(solution):
    global call_count
    call_count += 1

    # calcular iteración y gorrión dentro de la población
    iter_num = ((call_count - 1) // POP_SIZE) + 1
    sparrow_idx = ((call_count - 1) % POP_SIZE) + 1

    L = int(solution[0]) #  tamaño de la ventana
    num_layers = int(max(1, min(solution[1], MAX_LAYERS))) # numero de capas
    neuron_values = [int(solution[2 + i]) for i in range(MAX_LAYERS)] # numero de neurda
    dropout = float(solution[2 + MAX_LAYERS]) # apaga una cantidad de neuronas
    learning_rate = float(solution[3 + MAX_LAYERS]) #ajusta el peso de las neuronas
    batch_size = int(solution[4 + MAX_LAYERS]) # muestras antes de ajustar los pesos
    patience = int(solution[5 + MAX_LAYERS])# paciencia

# hiperparametros del modelo lestim
    params = {
        "lookback": L,
        "num_layers": num_layers,
        "num_neurons": neuron_values,
        "dropout": dropout,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "patience": patience
    }

    mse, _ = evaluate_model(params, iter_num=iter_num, sparrow_idx=sparrow_idx)
    return mse

# Límites de hiperparámetros

bounds = [IntegerVar(lb=24, ub=168), # ventana del tiempo
           IntegerVar(lb=1, ub=MAX_LAYERS)] # numerop de capas
for _ in range(MAX_LAYERS): # para cada capa
    bounds.append(IntegerVar(lb=32, ub=256)) # nuemro de neuronas
bounds += [FloatVar(lb=0.0, ub=0.5), # dropout
           FloatVar(lb=1e-4, ub=5e-3), #learning rate
           IntegerVar(lb=32, ub=128), # batch size
           IntegerVar(lb=10, ub=50)] # paciencia

# Optimización SSA

problem = Problem(
    obj_func=objective_function, # funcion que queremos minimizar
    bounds=bounds,  # limites para los hiper parametros
    minmax="min", # minimizar el valor devuelto por la funcion MSE
    name="LSTM_SSA" # NOMBRE DESCRIPTIVO
)

EPOCHS = 6
POP_SIZE = 6
# objeto de optimizacion 
ssa = SSA.OriginalSSA(EPOCHS, POP_SIZE) # iteraciones y gorriones
# inicia el proceso de optimizacion  y actualiza la poscion de los orriones seun el algortimo
best_solution = ssa.solve(problem) # la mejor solucion 

# los mejores valores encontrados
best_pos = best_solution.solution
# diccionario con estos valores
best_params = {
    "lookback": int(best_pos[0]),
    "num_layers": int(best_pos[1]),
    "num_neurons": [int(best_pos[2 + i]) for i in range(MAX_LAYERS)],
    "dropout": float(best_pos[2 + MAX_LAYERS]),
    "learning_rate": float(best_pos[3 + MAX_LAYERS]),
    "batch_size": int(best_pos[4 + MAX_LAYERS]),
    "patience": int(best_pos[5 + MAX_LAYERS])
}

# evalua el; modelo usando los mejores hiperparametros econtrados devuelve el mse y el mejor modelo
final_mse, best_model = evaluate_model(best_params, iter_num="final", sparrow_idx=0)

# tamaño de la ventana optimo
L_opt = int(best_params["lookback"])

# enviamos los valores de entrenamiento y test
X_train_final, y_train_final = create_windows(X_train_scaled, y_train_scaled, L_opt)
X_test_final, y_test_final = create_windows(X_test_scaled, y_test_scaled, L_opt)

# evalua el mejor modelo
best_model.eval()
with torch.no_grad():
    X_test_tensor = torch.tensor(X_test_final, dtype=torch.float32).to(device)
    # se predice con el mejor modelo
    y_pred_scaled = best_model(X_test_tensor).cpu().numpy().reshape(-1, 1)
# desnormalizar
y_pred_real = scaler_y.inverse_transform(y_pred_scaled).reshape(-1)
# desnormalizar
y_test_real = scaler_y.inverse_transform(y_test_final).reshape(-1)
# guardamnos las fechas der cada prediccion justo despues de la ventana inicial

fechas_test = df[fecha_col].iloc[split_idx + L_opt:split_idx + L_opt + len(y_test_real)].reset_index(drop=True)

# creamos un data ser con las predicciones y el valor real

df_pred = pd.DataFrame({
    "Fecha": fechas_test,
    f"{target_col}_real": y_test_real,
    f"{target_col}_predicho": y_pred_real,
    "Error": y_test_real - y_pred_real
})

mse = mean_squared_error(y_test_real, y_pred_real) # Error cuadrático medio
mae = mean_absolute_error(y_test_real, y_pred_real)  # Error absoluto medio
r2 = r2_score(y_test_real, y_pred_real) # coeficiente de determinacion
rmse = np.sqrt(mse) # Promedio de los errores


#graifico

plt.figure(figsize=(15, 6))
plt.plot(df_pred["Fecha"], df_pred[f"{target_col}_real"], label="Real", color="blue", linewidth=1.5)
plt.plot(df_pred["Fecha"], df_pred[f"{target_col}_predicho"], label="Predicho", color="orange", linewidth=1.5, alpha=0.8)
plt.title(f"Predicción de {target_col} - Modelo LSTM Multivariante + SSA\nR² = {r2:.4f} | RMSE = {rmse:.4f}", fontsize=14)
plt.xlabel("Fecha", fontsize=12)
plt.ylabel(f"{target_col} (µg/m³)", fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

output_path = r"C:\Users\aleja\OneDrive\Desktop\pytorch\predicciones_PM25_Multivariante_SSA.xlsx"
df_pred.to_excel(output_path, index=False)

df_history = pd.DataFrame(sparrow_history)
history_path = r"C:\Users\aleja\OneDrive\Desktop\pytorch\historial_gorriones_PM25_Multivariante.xlsx"
df_history.to_excel(history_path, index=False)

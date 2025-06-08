# ////////////////////////////////////////////////////////////////////////////////////////////////////////////
# //// PROBLEMA DE ESTIMACIÓN DE EDAD CON RADIOGRAFÍA MAXILOFACIAL
# //// REGRESIÓN CLÁSICA (UN SOLO VALOR PUNTUAL)
# //// DIVISIÓN DE LOS DATOS EN TRAIN, VALID Y TEST
# ////////////////////////////////////////////////////////////////////////////////////////////////////////////

# Biblioteca para aprendizaje profundo
import torch
import torchvision

# 
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import torch.nn as nn

#
if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA no está disponible. PyTorch no reconoce la GPU."
    )
device = "cuda"

#-------------------------------------------------------------------------------------------------------------

import os
working_dir = os.getcwd()
data_dir = working_dir + "/data/AE_maxillofacial/preprocessed/"
results_dir = working_dir + "/results/AE_maxillofacial/"
models_dir = working_dir + "/models/AE_maxillofacial/"

#-------------------------------------------------------------------------------------------------------------

# Manipulación de datos
import numpy as np
import pandas as pd

# Manejo y edición de imágenes
from PIL import Image

# Visualización de datos
import matplotlib.pyplot as plt
import seaborn as sns

# Operaciones aleatorias
import random

# Funciones matemáticas avanzadas
import math

# Evaluación y partición de modelos
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#
import time

#
from custom_models import ResNeXtRegressor

#-------------------------------------------------------------------------------------------------------------

# Creamos una semilla de aleatoriedad 
SEED = 23

# Fija la semilla para las operaciones aleatorias en Python puro
random.seed(SEED)

# Fija la semilla para las operaciones aleatorias en NumPy
np.random.seed(SEED)

# Fija la semilla para los generadores aleatorios de PyTorch en CPU
torch.manual_seed(SEED)

# Fija la semilla para todos los dispositivos GPU (todas las CUDA devices)
torch.cuda.manual_seed_all(SEED)

# Desactiva la autooptimización de algoritmos en cudnn, que puede introducir no determinismo
# torch.backends.cudnn.benchmark = False

# Fuerza a cudnn a usar operaciones determinísticas (más lento pero reproducible)
# torch.backends.cudnn.deterministic = True

# Obliga a Pytorch a usar algoritmos determinísticos cuando hay alternativa. Si no la hay, lanza un error.
# torch.use_deterministic_algorithms(True)

# Función auxiliar para asegurar que cada worker de DataLoader use una semilla basada en la global
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# Generador de números aleatorios para DataLoader
g = torch.Generator()
g.manual_seed(SEED)

#-------------------------------------------------------------------------------------------------------------


# Define las transformaciones aplicadas a las imágenes durante el entrenamiento en cada época.
# Estas transformaciones son aleatorias dentro de los rangos especificados, por lo que varían en cada época.
# - Redimensiona las imágenes a 448x224. Se ha escogido este tamaño dado que las imágenes son panorámicas y 
#   bastante maś anchas que altas.
# - (Regularización) Realiza un volteo horizontal a la mitad de las imágenes.
# - (Regularización) Aplica una rotación aleatoria de hasta +/-3 grados.
# - (Regularización) Aplica una transformación afín aleatoria con ligeras traslaciones (2%) y escalado (entre 
#   95% y 105%).
# - (Regularización) Modifica aleatoriamente el brillo y contraste para simular condiciones de iluminación 
#   variables.
# - Convierte la imagen a tensor, para que pueda ser manipulada por PyTorch.
# - Normaliza para ajustar la media y desviación típica de los canales RGB a los valores usados durante el 
#   entrenamiento en ImageNet.
train_transform = transforms.Compose(
    [transforms.Resize((448, 224)),
     transforms.RandomHorizontalFlip(p=0.5),
     transforms.RandomRotation(degrees=3),
     transforms.RandomAffine(degrees=0, translate=(0.02, 0.02), scale=(0.95, 1.05)), 
     transforms.ColorJitter(brightness=0.1, contrast=0.1), 
     transforms.ToTensor(),
     transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))]
)

# Define las transformaciones para las imágenes de validación y test, que son iguales que para entrenamiento 
# pero sin regularización
valid_transform = test_transform = transforms.Compose(
    [transforms.Resize((448, 224)),
     transforms.ToTensor(),
     transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))]
) 


# Define la clase MaxillofacialXRayDataset, que se utiliza para cargar imágenes de rayos X maxilofaciales 
# junto con su edad correspondiente desde un archivo de metadatos. Permite aplicar transformaciones a las 
# imágenes si se especifican.
class MaxillofacialXRayDataset(Dataset):
    
    def __init__(self, metadata_file, images_dir, transform=None):
        """
        metadata_file: Ruta al archivo de metadatos (CSV u otro formato)
        images_dir: Ruta a la carpeta de imágenes (entrenamiento o prueba)
        transform: Transformaciones a aplicar a las imágenes (normalización, etc.)
        """
        self.metadata = pd.read_csv(metadata_file)  # Cargar los metadatos
        self.images_dir = images_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.metadata)
        
    def __getitem__(self, idx):
        # Obteniene el nombre de la imagen y su valor desde los metadatos
        img_name = os.path.join(self.images_dir, self.metadata.iloc[idx]['ID'])  # Ajusta según la estructura
        target = self.metadata.iloc[idx]['Age'].astype(np.float32)  # Ajusta según el formato de tus metadatos
        
        # Abre la imagen
        image = Image.open(img_name)
        
        # Aplica transformaciones si es necesario
        if self.transform:
            image = self.transform(image)
        
        return image, target
    
# --------------------------------------------------------------------

# Crea el Dataset de entrenamiento con augmentations
trainset = MaxillofacialXRayDataset(
    metadata_file=data_dir + 'metadata_train.csv',
    images_dir=data_dir + 'train/',
    transform=train_transform
)

# Crea el Dataset de validación con solo resize y normalización 
validset = MaxillofacialXRayDataset(
    metadata_file=data_dir + 'metadata_train.csv',  
    images_dir=data_dir + 'train/',               
    transform=valid_transform                       
)

# Crea el Dataset de test con solo resize y normalización
testset  =  MaxillofacialXRayDataset(
    metadata_file = data_dir + 'metadata_test.csv',
    images_dir = data_dir + 'test/',
    transform = test_transform
) 

# --------------------------------------------------------------------

# Establece un batch size de 32 
BATCH_SIZE = 32

# Obtiene las edades enteras del trainset
intAges = np.floor(trainset.metadata['Age'].astype(float).to_numpy()).astype(int)
# Como se ha visto antes, hay una única instancia con edad 26, que el algoritmo de separación de entrenamiento 
# y validación será incapaz de dividir de forma estratificada. Para evitar el error, reasigna esa instancia a 
# la edad inmediatamente inferior
intAges[intAges==26]=25

# Divide el conjunto de datos completo de entrenamiento en dos subconjuntos de forma estratificada:
# - Entrenamiento (80% de las instancias)
# - Validación (20% de las instancias)
train_indices, valid_indices =  train_test_split(
    range(len(trainset)),
    train_size=0.80,
    shuffle=True,
    stratify=intAges
)
train_subset = Subset(trainset, train_indices)
valid_subset = Subset(validset, valid_indices)

# Crea DataLoader de entrenamiento
train_loader =  DataLoader(
    train_subset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=2, 
    pin_memory=True, 
    worker_init_fn=seed_worker,
    generator=g,
)

# Crea DataLoader de validación
valid_loader =  DataLoader(
    valid_subset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=2, 
    pin_memory=True,
    worker_init_fn=seed_worker,
    generator=g,
)

# Crea DataLoader de test
test_loader = DataLoader(
    testset, 
    batch_size=BATCH_SIZE, 
    shuffle=False
)

print("✅ Datasets de imágenes cargados\n")

#-------------------------------------------------------------------------------------------------------------

#
model = ResNeXtRegressor().to(device)

print("✅ Modelo cargado\n")

#-------------------------------------------------------------------------------------------------------------

def train(model, dataloader, loss_fn, optimizer, scheduler=None, device="cuda"):
    
    # Pone la red en modo entrenamiento (esto habilita el dropout)
    model.train()  
    
    # Inicializa la pérdida acumulada para esta época
    epoch_loss = 0

    # Itera sobre todos los lotes de datos del DataLoader
    for inputs, targets in dataloader:
        
        # Obtiene las imágenes de entrenamiento y sus valores objetivo
        inputs, targets = inputs.to(device), targets.to(device)

        # Limpia los gradientes de la iteración anterior
        optimizer.zero_grad()           
        
        # Pasa las imágenes de entrada a través de la red (propagación hacia adelante)
        outputs = model(inputs)       
        
        # Calcula la pérdida de las predicciones
        loss = loss_fn(outputs, targets) 
        
        # Realiza la retropropagación para calcular los gradientes (propagación hacia atrás)
        loss.backward()
        
        # Actualiza los parámetros del modelo
        optimizer.step()            
        
        # Actualiza el scheduler de la tasa de aprendizaje (si se proporciona)
        if scheduler is not None:
            scheduler.step()   
 
        # Acumula la pérdida de este batch
        epoch_loss += loss.item()        
    
    # Calcula la pérdida promedio de la época y la devolvemos
    avg_loss = epoch_loss / len(dataloader)
    return avg_loss

#-------------------------------------------------------------------------------------------------------------

def inference(model, dataloader, metric_fn=None, device="cuda"):
    
    # Pone la red en modo evaluación (esto desactiva el dropout)
    model.eval()  
    
    # Inicializa listas para almacenar las predicciones y los valores objetivo (target)
    all_predicted = []
    all_targets = []
    
    # No calcula los gradientes durante la validación para ahorrar memoria y tiempo
    with torch.no_grad():
        
        # Itera sobre el conjunto a evaluar
        for inputs, targets in dataloader:
            
             # Obtiene las imágenes de validación y sus valores objetivo
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Realiza una predicción con el modelo
            outputs = model(inputs)
            
            # Almacena las predicciones y los targets
            all_predicted.append(outputs.cpu())
            all_targets.append(targets.cpu())

    # Concatena todas las predicciones y targets
    all_predicted = torch.cat(all_predicted)
    all_targets = torch.cat(all_targets)

    if metric_fn is None:
        return all_predicted, all_targets

    # Aplica la función de métrica y la devuelve
    metric_value = metric_fn(all_predicted, all_targets)
    return metric_value
    
#-------------------------------------------------------------------------------------------------------------

# Ruta donde se guardará el modelo con mejor desempeño
best_model_path = models_dir + "AE_model01_base.pth" 

# Define la función de pérdida a usar
criterion = nn.MSELoss().to(device)

# Establece el learning rate base y weight decay 
base_lr = 3e-2
wd = 2e-4

#-------------------------------------------------------------------------------------------------------------

# Congela los parámetros del extractor de características
for param in model.feature_extractor.parameters():
    param.requires_grad = False

# Configura el optimizador para el entrenamiento de la nueva cabecera (el módulo classifier)
optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=base_lr, weight_decay=wd)

# Numero de épocas que se entrena la nueva cabecera
NUM_EPOCHS_HEAD = 2

for epoch in range(NUM_EPOCHS_HEAD):    

    # Entrena el modelo con el conjunto de entrenamiento
    head_train_loss = train(model, train_loader, criterion, optimizer, device=device)

    # Evalua el modelo con el conjunto de validación
    head_valid_loss = inference(model, valid_loader, criterion, device=device)

    # Imprime los valores de pérdida obtenidos en entrenamiento y validación 
    print(f'Epoch {epoch+1} | Train Loss: {head_train_loss:.3f} | Validation Loss: {head_valid_loss:.3f}')

# Guarda los pesos del modelo actual como los mejores hasta ahora
torch.save(model.state_dict(), best_model_path)

print("✅ Entrenamiento de la nueva cabecera completado\n")

#-------------------------------------------------------------------------------------------------------------

# Descongela todos los parámetros del modelo
for param in model.parameters():
    param.requires_grad = True

# Número máximo de épocas a entrenar (si no se activa el early stopping)
MAX_EPOCHS = 100

# Número mínimo de épocas a entrenar
MIN_EPOCHS = 30

# Número de épocas sin mejora antes de detener el entrenamiento
PATIENCE = 10

# Inicializa la mejor pérdida de validación como la obtenida en el entrenamiento de la cabecera
best_valid_loss = head_valid_loss 

# Contador de épocas sin mejora
epochs_no_improve = 0 

# Crea una lista para almacenar los nombres de las capas del modelo
layer_names = []
for (name, param) in model.named_parameters():
    layer_names.append(name)

# Establece las reglas para el learning rate discriminativo   
lr_div = 100            # Factor de reducción entre el learning rate más alto y el más pequeño
max_lr = base_lr/2      # Learning rate más alto (capa más superficial)
min_lr = max_lr/lr_div  # Learning rate más bajo (capa más profunda) 

# Obtenemos los grupos de capas del modelo (de más superficiales a más profundas)
layer_groups = model.get_layer_groups()
n_layers = len(layer_groups)

# Calcula un learning rate distinto para cada grupo de capas, interpolando exponencialmente entre min_lr 
# (capas profundas) y max_lr (capas superficiales)
lrs = [
    min_lr * (max_lr / min_lr) ** (i / (n_layers - 1))
    for i in range(n_layers)
]

# Lista en la que se almacenarán los parámetros por grupo y sus lr
param_groups = []
for layer_group, lr in zip(layer_groups, lrs):
    param_groups.append(
        {"params": layer_group, "lr": lr}
    )

# Configura el optimizador con los hiperparámetros escogidos
optimizer = torch.optim.AdamW(param_groups, lr=base_lr, weight_decay=wd)

# Crea el scheduler OneCycleLR
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, 
    max_lr=lrs, 
    steps_per_epoch=len(train_loader),
    epochs=MAX_EPOCHS,
    pct_start=MIN_EPOCHS/MAX_EPOCHS*0.8
)

# Listas para almacenar las pérdidas de entrenamiento y validación
train_losses = []
valid_losses = []

# Temporizador de inicio
start_time = time.time()

# Bucle de entrenamiento por épocas
for epoch in range(MAX_EPOCHS):
    
    # Entrena el modelo con el conjunto de entrenamiento
    train_loss = train(model, train_loader, criterion, optimizer, scheduler, device)
    train_losses.append(train_loss)
    
    # Evalua el modelo con el conjunto de validación
    valid_loss = inference(model, valid_loader, criterion, device)
    valid_losses.append(valid_loss)
    
    # Imprime los valores de pérdida obtenidos en entrenamiento y validación  
    print(f'Epoch {epoch+1} | Train Loss: {train_loss:.3f} | Validation Loss: {valid_loss:.3f}')
    
    # Comprueba si la pérdida en validación ha mejorado
    if valid_loss < best_valid_loss:
        
        # Actualiza la mejor pérdida en validación obtenida hasta ahora
        best_valid_loss = valid_loss
        
        # Reinicia el contador de épocas sin mejora si la pérdida ha mejorado
        epochs_no_improve = 0
        
        # Guarda los pesos del modelo actual como los mejores hasta ahora
        torch.save(model.state_dict(), best_model_path)
        
    else:
        # Incrementa el contador si no hay mejora en la pérdida de validación
        epochs_no_improve += 1

    # Si no hay mejora durante un número determinado de épocas (patience) y ya ha pasado el número mínimo de 
    # épocas, detiene el entrenamiento
    if epochs_no_improve >= PATIENCE and (epoch+1) > MIN_EPOCHS: 
        print(f'Early stopping at epoch {epoch+1}')
        break
    
    
# Carga los pesos del modelo que obtuvo la mejor validación
model.load_state_dict(torch.load(best_model_path))

# Cálculo de tiempo total de entrenamiento 
end_time = time.time()
elapsed_time = end_time - start_time

# Convierte el tiempo en horas, minutos y segundos
hours = int(elapsed_time // 3600)
minutes = int((elapsed_time % 3600) // 60)
seconds = int(elapsed_time % 60)

# Imprime el tiempo de ejecución en formato horas:minutos:segundos
print(f"\nEl entrenamiento y validación ha tardado {hours} horas, {minutes} minutos y {seconds} segundos.")

# Grafica las curvas de aprendizaje
plt.figure(figsize=(8, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(valid_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
plt.grid(True)

# Guarda la imagen
plt.savefig(results_dir + 'learning_curve_AE_model01_base.png', dpi=300, bbox_inches='tight')  

print("✅ Entrenamiento de la red completa completado\n")

#-------------------------------------------------------------------------------------------------------------

# Obtiene los valores predichos y los verdaderos 
test_pred_values, test_true_values = inference(model, test_loader)

# Calcula el MAE 
test_mae = torch.mean(torch.abs(test_true_values - test_pred_values))
print(f'Error Absoluto Medio (MAE) en test: {test_mae:>.3f}')

# Calcula e imprime el MSE
test_mse = torch.mean((test_true_values - test_pred_values) ** 2)
print(f'Error Cuadrático Medio (MSE) en test: {test_mse:>.3f}')

# Calcula e imprime el R²
r2 = r2_score(test_true_values, test_pred_values)
print(f"R² en test: {r2:.4f}")

print("✅ Testeo de la red completado\n")


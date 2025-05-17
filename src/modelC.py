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

#-------------------------------------------------------------------------------------------------------------------------

import os
working_dir = os.getcwd()
data_dir = working_dir + "/data/AE_maxillofacial/preprocessed/"
results_dir = working_dir + "/results/AE_maxillofacial/"
models_dir = working_dir + "/models/AE_maxillofacial/"

#-------------------------------------------------------------------------------------------------------------------------

# Manipulación de datos
import numpy as np
import pandas as pd

# Manejo y edición de imágenes
from PIL import Image

# Resumen de modelos en PyTorch
from torchsummary import summary

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

#-------------------------------------------------------------------------------------------------------------------------

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

#-------------------------------------------------------------------------------------------------------------------------

# Definimos las transformaciones que prepara las imágenes para ser procesadas por el modelo:
# - Redimensiona las imágenes a 448x224. Se ha escogido este tamaño dado que las imágenes son panorámicas y bastante 
#   maś anchas que altas.
# - Convertir la imagen a tensor, para que pueda ser manipulada por PyTorch.
# - Normalizar para ajustar la media y desviación típica de los canales RGB a los valores usados durante el entrenamiento 
#   en ImageNet.
# ...
train_transform = transforms.Compose(
    [transforms.Resize((448, 224)),
     transforms.RandomHorizontalFlip(p=0.5),
     transforms.RandomRotation(degrees=3),
     transforms.RandomAffine(degrees=0, translate=(0.02, 0.02), scale=(0.95, 1.05)), 
     transforms.ColorJitter(brightness=0.1, contrast=0.1), 
     transforms.ToTensor(),
     transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))]
)

val_transform = test_transform = transforms.Compose(
    [transforms.Resize((448, 224)),
     transforms.ToTensor(),
     transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))]
) 

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
        """
        Obtener una imagen y su etiqueta por índice.
        """
        # Obtener el nombre de la imagen y su valor desde los metadatos
        img_name = os.path.join(self.images_dir, self.metadata.iloc[idx]['ID'])  # Ajusta según la estructura
        target = float(self.metadata.iloc[idx]['Age'])  # Ajusta según el formato de tus metadatos
        
        # Abrir la imagen
        image = Image.open(img_name)
        
        # Aplicar transformaciones si es necesario
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
    transform=val_transform                       
)

# Crea el Dataset de test con solo resize y normalización
testset  =  MaxillofacialXRayDataset(
    metadata_file = data_dir + 'metadata_test.csv',
    images_dir = data_dir + 'test/',
    transform = test_transform
) 

# --------------------------------------------------------------------

#
BATCH_SIZE = 64

# Obtenemos las edades enteras del trainset
intAges = np.floor(trainset.metadata['Age'].astype(float).to_numpy()).astype(int)
# Como vimos antes, hay una única instancia con edad 26, que el algoritmo de separación de entrenamiento 
# y validación será incapaz de dividir de forma estratificada. Para evitar el error, reasignamos esa 
# instancia a la edad inmediatamente inferior.
intAges[intAges==26]=25


# Dividimos el conjunto de datos de forma estratificada en dos subconjuntos:
# - Entrenamiento (80% de las instancias)
# - Validación (20% de las instancias)

# 1) Dividimos el conjunto de datos de entrenamiento y validación
train_indices, valid_indices =  train_test_split(
    range(len(trainset)),
    train_size=0.80,
    shuffle=True,
    stratify=intAges
)

# 2) Creamos los subconjuntos para entrenamiento, validación y calibración
train_subset = Subset(trainset, train_indices)
valid_subset = Subset(validset, valid_indices)

# 3) Creamos DataLoaders para cada subconjunto
train_loader =  DataLoader(
    train_subset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=2, 
    pin_memory=True, 
    worker_init_fn=seed_worker,
    generator=g,
)

valid_loader =  DataLoader(
    valid_subset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=2, 
    pin_memory=True,
    worker_init_fn=seed_worker,
    generator=g,
)

# Crear el DataLoader para test
test_loader = DataLoader(
    testset, 
    batch_size=BATCH_SIZE, 
    shuffle=False
)

print("✅ Datasets de imágenes cargados")

#-------------------------------------------------------------------------------------------------------------------------

class FeatureExtractorResNeXt(nn.Module):
    
    def __init__(self):
        
        super(FeatureExtractorResNeXt, self).__init__()
        
        resnext = torchvision.models.resnext50_32x4d(weights='DEFAULT')
        
        self.layer0 = nn.Sequential(
            resnext.conv1,
            resnext.bn1,
            resnext.relu,
            resnext.maxpool,
        )
        self.layer1 = resnext.layer1
        self.layer2 = resnext.layer2
        self.layer3 = resnext.layer3
        self.layer4 = resnext.layer4
        
        
    def forward(self, x):
        
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        return x
        

#
class ResNeXtRegressor(nn.Module):
    
    def __init__(self):
        
        super(ResNeXtRegressor, self).__init__()
        
        self.feature_extractor = FeatureExtractorResNeXt()
        
        # Nueva head
        self.pool_avg = nn.AdaptiveAvgPool2d((1,1))
        self.pool_max = nn.AdaptiveMaxPool2d((1, 1))
        self.flatten = nn.Flatten()
        
        fc1 = nn.Sequential(
            nn.BatchNorm1d(4096),  # 2048 (avg) + 2048 (max)
            nn.Dropout(p = 0.5),
            nn.Linear(4096, 512),
            nn.ReLU(inplace=True)
        )
        
        fc2 = nn.Sequential(
            nn.BatchNorm1d(512),
            nn.Dropout(p = 0.5),
            nn.Linear(512, 1) 
        )
        
        self.regressor = nn.Sequential(
            fc1, fc2
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        avg = self.pool_avg(x)
        max = self.pool_max(x)
        x = torch.cat([avg, max], dim=1) 
        x = self.flatten(x)
        x = self.regressor(x)
        return x


#
model = ResNeXtRegressor().to(device)

print("✅ Modelo cargado")

#-------------------------------------------------------------------------------------------------------------------------

def train(model, dataloader, loss_fn, optimizer, scheduler=None, device="cuda"):
    
    # Ponemos la red en modo entrenamiento (esto habilita el dropout)
    model.train()  
    
    # Inicializamos la pérdida acumulada para esta época
    epoch_loss = 0

    # Iteramos sobre todos los lotes de datos del DataLoader
    for inputs, targets in dataloader:
        
        # Obtenemos las imágenes de entrenamiento y sus valores objetivo
        inputs, targets = inputs.to(device), targets.to(device).float().unsqueeze(1)

        # Limpiamos los gradientes de la iteración anterior
        optimizer.zero_grad()           
        
        # Pasamos las imágenes de entrada a través de la red (propagación hacia adelante)
        outputs = model(inputs)       
        
        # Calculamos la pérdida de las predicciones
        loss = loss_fn(outputs, targets) 
        
        # Realizamos la retropropagación para calcular los gradientes (propagación hacia atrás)
        loss.backward()
        
        # Actualizamos los parámetros del modelo
        optimizer.step()            
        
        #
        if scheduler is not None:
            scheduler.step()   
 
        # Acumulamos la pérdida de este batch
        epoch_loss += loss.item()        
    
    # Calculamos la pérdida promedio de la época y la devolvemos
    avg_loss = epoch_loss / len(dataloader)
    return avg_loss

#-------------------------------------------------------------------------------------------------------------------------

def evaluate(model, dataloader, metric_fn=None, device="cuda"):
    
    # Ponemos la red en modo evaluación (esto desactiva el dropout)
    model.eval()  
    
    # Inicializamos listas para almacenar las predicciones y los valores objetivo (target)
    all_predicted = []
    all_targets = []
    
    # No calculamos los gradientes durante la validación para ahorrar memoria y tiempo
    with torch.no_grad():
        
        # Iteramos sobre el conjunto a evaluar
        for inputs, targets in dataloader:
            
             # Obtenemos las imágenes de validación y sus valores objetivo
            inputs, targets = inputs.to(device), targets.to(device).float().unsqueeze(1)
            
            # Realizamos una predicción con el modelo
            outputs = model(inputs)
            
            # Almacenamos las predicciones y los targets
            all_predicted.append(outputs.cpu())
            all_targets.append(targets.cpu())

    # Concatenamos todas las predicciones y targets
    all_predicted = torch.cat(all_predicted)
    all_targets = torch.cat(all_targets)

    if metric_fn is None:
        return all_predicted, all_targets

    # Aplicamos la función de métrica y la devolvemos 
    metric_value = metric_fn(all_predicted, all_targets)
    return metric_value
    
#-------------------------------------------------------------------------------------------------------------------------

# Ruta donde se guardará el modelo con mejor desempeño
best_model_path = models_dir + "modelC.pth" 

# Definimos la función de pérdida a usar
criterion = nn.MSELoss()

#
base_lr = 1.5e-2
wd = 1e-4

# Configuramos el optimizador para ...
optimizer = torch.optim.AdamW(model.regressor.parameters(), lr=base_lr, weight_decay=wd)

# Congelamos los parámetros del extractor de características
for param in model.feature_extractor.parameters():
    param.requires_grad = False

#
EPOCHS_PRETRAIN = 5

for epoch in range(EPOCHS_PRETRAIN):
    
    # Entrenamos el modelo con el conjunto de entrenamiento
    train_loss = train(model, train_loader, criterion, optimizer, device=device)
    
    # Evaluamos el modelo con el conjunto de validación
    valid_loss = evaluate(model, valid_loader, criterion, device=device)
    
    # Añadimos las marcas de pérdidas en entrenamiento y validación a las listas y las imprimimos 
    print(f'Epoch {epoch+1} | Train Loss: {train_loss:.2f} | Validation Loss: {valid_loss:.2f}')


print("✅ Preentrenamiento de la nueva cabecera realizado")

#-------------------------------------------------------------------------------------------------------------------------

# Descongelamos todos los parámetros del modelo
for param in model.parameters():
    param.requires_grad = True

# Número máximo de épocas a entrenar (si no se activa el early stopping)
MAX_EPOCHS = 40

# Número de épocas sin mejora antes de detener el entrenamiento
PATIENCE = 12

# Inicializamos la mejor pérdida de validación como infinito (para encontrar el mínimo)
best_valid_loss = float('inf')   

# Contador de épocas sin mejora
epochs_no_improve = 0  

# Establecemos las reglas para el learning rate discriminativo               
lr_mult = 100                  
max_lr = base_lr/2              # Learning rate más alto, que se usará en las capas más superficiales 
min_lr = max_lr/lr_mult         # Learning rate más bajo, que se usará en las capas más profundas 

# Generamos una lista con los learning rates espaciados linealmente entre min_lr y max_lr
n_layers = 6
lrs = torch.linspace(min_lr, max_lr, n_layers)

# Creamos una lista donde guardar los parámetros del modelo por capas (de menor a mayor profundidad)
param_groups = []
param_groups.append({'params': model.feature_extractor.layer0.parameters(), 'lr':lrs[0].item()})
param_groups.append({'params': model.feature_extractor.layer1.parameters(), 'lr':lrs[1].item()})
param_groups.append({'params': model.feature_extractor.layer2.parameters(), 'lr':lrs[2].item()})
param_groups.append({'params': model.feature_extractor.layer3.parameters(), 'lr':lrs[3].item()})
param_groups.append({'params': model.feature_extractor.layer4.parameters(), 'lr':lrs[4].item()})
param_groups.append({'params': model.regressor.parameters(), 'lr':lrs[5].item()})

# Configuramos el optimizador con los hiperparámetros escogidos
optimizer = torch.optim.AdamW(param_groups, lr=base_lr, weight_decay=wd)

# Crea el scheduler OneCycleLR
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, 
    max_lr=lrs.tolist(), 
    steps_per_epoch=len(train_loader),
    epochs=MAX_EPOCHS
)

# Listas para almacenar las pérdidas de entrenamiento y validación
train_losses = []
valid_losses = []

# Temporizador de inicio
start_time = time.time()

# Bucle de entrenamiento por épocas
for epoch in range(MAX_EPOCHS):
    
    # Entrenamos el modelo con el conjunto de entrenamiento
    train_loss = train(model, train_loader, criterion, optimizer, scheduler, device)
    train_losses.append(train_loss)
    
    # Evaluamos el modelo con el conjunto de validación
    valid_loss = evaluate(model, valid_loader, criterion, device)
    valid_losses.append(valid_loss)
    
    # Añadimos las marcas de pérdidas en entrenamiento y validación a las listas y las imprimimos 
    print(f'Epoch {epoch+1} | Train Loss: {train_loss:.2f} | Validation Loss: {valid_loss:.2f}')
    
    # Comprobamos si la pérdida en validación ha mejorado
    if valid_loss < best_valid_loss:
        
        # Actualizamos la mejor pérdida en validación obtenida hasta ahora
        best_valid_loss = valid_loss
        
        # Reiniciamos el contador de épocas sin mejora si la pérdida ha mejorado
        epochs_no_improve = 0
        
        # Guardamos los pesos del modelo actual como los mejores hasta ahora
        torch.save(model.state_dict(), best_model_path)
        
    else:
        # Incrementamos el contador si no hay mejora en la pérdida de validación
        epochs_no_improve += 1

    # Si no hay mejora durante un número determinado de épocas (patience), detenemos el entrenamiento
    if epochs_no_improve >= PATIENCE:
        print(f'Early stopping at epoch {epoch+1}')
        break
    

#Cargamos los pesos del modelo que obtuvo la mejor validación
model.load_state_dict(torch.load(best_model_path))

# Cálculo de tiempo total de entrenamiento 
end_time = time.time()
elapsed_time = end_time - start_time

# Convertir el tiempo en horas, minutos y segundos
hours = int(elapsed_time // 3600)
minutes = int((elapsed_time % 3600) // 60)
seconds = int(elapsed_time % 60)

# Imprime el tiempo de ejecución en formato horas:minutos:segundos
print(f"\nEl entrenamiento y validación ha tardado {hours} horas, {minutes} minutos y {seconds} segundos.")

print("✅ Entrenamiento de la red completa realizado")

#-------------------------------------------------------------------------------------------------------------------------

# Imprime las listas de pérdidas
print("Train losses: ", train_losses)
print("Validation losses: ", valid_losses)

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
plt.savefig(results_dir + 'train_modelC.png', dpi=300, bbox_inches='tight')  

#-------------------------------------------------------------------------------------------------------------------------

# Obtiene los valores predichos y los verdaderos 
test_pred_values, test_true_values = evaluate(model, test_loader)

# Calcula el MAE 
test_mae = torch.mean(torch.abs(test_true_values - test_pred_values))
print(f'Error Absoluto Medio (MAE) en test: {test_mae:.2f}')

# Calcula e imprime el MSE
test_mse = torch.mean((test_true_values - test_pred_values) ** 2)
print(f'Error Cuadrático Medio (MSE) en test: {test_mse:.2f}')

# Calcula e imprime el R²
r2 = r2_score(test_true_values, test_pred_values)
print(f"R² en test: {r2:.4f}")


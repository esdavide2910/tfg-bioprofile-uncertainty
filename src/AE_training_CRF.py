# ////////////////////////////////////////////////////////////////////////////////////////////////////////////
# //// PROBLEMA DE ESTIMACIÓN DE EDAD CON RADIOGRAFÍA MAXILOFACIAL
# ////////////////////////////////////////////////////////////////////////////////////////////////////////////

# # Biblioteca para aprendizaje profundo
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
device = 'cuda'

#-------------------------------------------------------------------------------------------------------------

import os
working_dir = os.getcwd()
data_dir = working_dir + '/data/AE_maxillofacial/preprocessed/'

#-------------------------------------------------------------------------------------------------------------

# Manejo del sistema y argumentos de línea de comandos
import sys
import argparse

# Control de advertencias
import warnings

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

# Manejo de tiempo y fechas
import time

# Modelos y funciones de pérdida personalizados 
from custom_models import ResNeXtRegressor, QuantileLoss, QuantileSquaredLoss
from coverage_metrics import empirical_coverage, mean_interval_size

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
# CONFIGURACIÓN DE ARGUMENTOS PARA ENTRENAMIENTO Y EVALUACIÓN DE LOS MODELOS 

# Crear el parser
parser = argparse.ArgumentParser(description="Procesa algunos argumentos.")

# Funciones auxiliares
def validate_confidence(x):
    x = float(x)
    if not 0.0 <= x <= 1.0:
        raise argparse.ArgumentTypeError(f"El nivel de confianza debe estar entre 0 y 1 (se recibió {x})")
    return round(x, 2)

def validate_file_extension(filename, extensions, arg_name):
    if not filename.lower().endswith(extensions):
        raise argparse.ArgumentTypeError(
            f"El archivo para '{arg_name}' debe tener extensión: {', '.join(extensions)}"
        )
    return filename

def output_stream_type(filename):
    # Permite pasar "-" como sinónimo de stdout
    if filename == "-":
        return sys.stdout
    return filename  # devolvemos filename, lo abriremos luego en el modo correcto


# Argumento 1: Tipo de modelo predictivo
pred_model_types = ['base', 'QR', 'ICP', 'CRF', 'CQR']
parser.add_argument(
    '-t', '--pred_model_type',
    type=str,
    required=True,
    choices=pred_model_types,
    help="Tipo de modelo predictivo a utilizar. Opciones: " + ", ".join(pred_model_types)
)

# Argumento 2: Nivel de confianza
parser.add_argument(
    '-c', '--confidence', 
    type=validate_confidence,
    default=0.9,
    help="Nivel de confianza para intervalos de predicción (0.0 a 1.0). Valor por defecto: 0.9"
)

# Argumento 3: Ruta del modelo entrenado
parser.add_argument(
    '-m', '--model_path',
    type=str, 
    required=True,
    help="Archivo de destino para guardar el modelo entrenado (formato .pth)"
)

# Argumento 4: Salida de resultados
parser.add_argument(
    '-o', '--output_stream',
    type=output_stream_type,
    default=sys.stdout,
    help="Archivo para redirigir TODA la salida (stdout + stderr). Usa '-' para terminal (por defecto)"
)

# Argumento 5: Append a la salida
parser.add_argument(
    '--append_output',
    action='store_true',
    help="Si se especifica, añade la salida al archivo en vez de sobrescribirlo."
)

# Argumento 6: Visualización de entrenamiento
parser.add_argument(
    '-g', '--training_plot',
    type=argparse.FileType('wb'),
    default=None,
    help="Archivo para guardar gráfico de curva de aprendizaje (.png, .jpg, .jpeg)"
)

# Argumento 7: Ignora los warnings
parser.add_argument('--ignore_warnings', action='store_true',
                    help="Ignora todos los warnings durante la ejecución")


# Parsear los argumentos
args = parser.parse_args()

# Validaciones adicionales
try:
    # Validación de extensión para el modelo
    validate_file_extension(args.model_path, ('.pth',), 'model_path')
    
    # Validación de extensión para el gráfico si se especifica
    if args.training_plot:
        validate_file_extension(args.training_plot.name, ('.png', '.jpg', '.jpeg'), 'training_plot')

    # Apertura segura del stream de salida
    if isinstance(args.output_stream, str):
        mode = 'a' if args.append_output else 'w'
        args.output_stream = open(args.output_stream, mode=mode, encoding='utf-8')
        sys.stdout = args.output_stream
        sys.stderr = args.output_stream
        
    # Configurar warnings
    if args.ignore_warnings:
        warnings.filterwarnings('ignore')
        
except Exception as e:
    parser.error(str(e))
    
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
# junto con su edad correspondiente desde un fichero de metadatos. 
class MaxillofacialXRayDataset(Dataset):
    
    def __init__(self, metadata_file, images_dir, transform=None):
        """
        metadata_file: Ruta al fichero de metadatos (CSV u otro formato)
        images_dir: Ruta al directorio de imágenes (entrenamiento o prueba)
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
    
# ------------------------------------------------------------------------------------------------------------

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

# ------------------------------------------------------------------------------------------------------------

# Establece un batch size de 32 
BATCH_SIZE = 32

# Obtiene las edades enteras del trainset
intAges = np.floor(trainset.metadata['Age'].astype(float).to_numpy()).astype(int)
# Hay una única instancia con edad 26, que el algoritmo de separación de entrenamiento y validación será 
# incapaz de dividir de forma estratificada. Para evitar el error, reasigna esa instancia a la edad 
# inmediatamente inferior
intAges[intAges==26]=25

# Función auxiliar para crear un DataLoader a partir de un subconjunto del dataset
def create_loader(dataset, indices):
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, 
                      pin_memory=True, worker_init_fn=seed_worker, generator=g)
 
# División para los modelos 'base' y 'QR'
if args.pred_model_type in ['base','QR']:

    # Divide el conjunto de datos completo de entrenamiento en dos subconjuntos de forma estratificada:
    # - Entrenamiento (80% de las instancias)
    # - Validación (20% de las instancias)

    train_idx, valid_idx =  train_test_split(
        range(len(trainset)), train_size=0.80, shuffle=True, stratify=intAges
    )

    train_loader = create_loader(trainset, train_idx)
    valid_loader = create_loader(validset, valid_idx)

# División para los modelos 'ICP' y 'CQR'
else: 
    
    # Divide el conjunto de datos completo de entrenamiento en tres subconjuntos de forma estratificada:
    # - Entrenamiento (68% de las instancias)
    # - Validación (17% de las instancias)
    # - Calibración (15% de las instancias)

    train_idx, calib_idx = train_test_split(
        range(len(trainset)), train_size=0.85, shuffle=True, stratify=intAges
    )

    train_idx, valid_idx = train_test_split(
        train_idx, train_size=0.8, shuffle=True, stratify=[intAges[i] for i in train_idx]
    )

    train_loader = create_loader(trainset, train_idx)
    valid_loader = create_loader(validset, valid_idx)
    calib_loader = create_loader(validset, calib_idx)

# Crea DataLoader de test
test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

print("✅ Datasets de imágenes cargados\n")

#-------------------------------------------------------------------------------------------------------------
# CARGA DE MODELO Y DEFINICIÓN DE FUNCIÓN DE PÉRDIDA

# Si el modelo tiene un intervalo de salida (mide de alguna forma la incertidumbre), calcula alpha a partir
# del nivel de confianza
if args.pred_model_type in ['QR', 'ICP', 'CRF', 'CQR']:
    alpha = (1-args.confidence)

# 
if args.pred_model_type in ['base', 'ICP', 'CRF']:

    # Inicializa el modelo de regresión estándar con una sola salida
    model = ResNeXtRegressor().to(device)

    # Define la función de pérdida a usar
    criterion = nn.MSELoss().to(device)

# ... para 'QR' y 'CQR'
else:
    
    # Define los cuantiles que el modelo debe predecir (p.ej., 0.05 y 0.95 para 90% de confianza)
    quantiles = [alpha/2, 0.5, 1-alpha/2]

    # Inicializa el modelo con múltiples salidas, una por cada cuantil
    model = ResNeXtRegressor(len(quantiles)).to(device)

    # Define la función de pérdida a usar
    criterion = QuantileSquaredLoss(quantiles).to(device)


print("✅ Modelo cargado\n")

#-------------------------------------------------------------------------------------------------------------
# FUNCIONES DE ENTRENAMIENTO (DE UNA ÉPOCA), INFERENCIA y EVALUACIÓN 

def train(model, dataloader, loss_fn, optimizer, scheduler=None, device='cuda'):
    
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


# def inference(model, dataloader, device='cuda'):
    
#     # Pone la red en modo evaluación
#     model.eval()  
    
#     # Inicializa listas para almacenar los valores objetivo y las predicciones
#     all_targets = []
#     all_outputs = []
    
#     # No calcula los gradientes durante la validación para ahorrar memoria y tiempo
#     with torch.no_grad():
        
#         # Itera sobre el conjunto a evaluar
#         for inputs, targets in dataloader:
            
#             # Obtiene las imágenes y sus valores objetivo
#             inputs, targets = inputs.to(device), targets.to(device)
#             all_targets.append(targets.cpu())
            
#             # Realiza predicciones con el modelo y las almacena
#             outputs = model(inputs)
                
#             all_outputs.append(outputs.cpu())

#     # Concatena todas las predicciones y targets, y los devuelve
#     all_targets = torch.cat(all_targets)
#     all_outputs = torch.cat(all_outputs)
    
#     return all_targets, all_outputs

def inference(
    model, dataloader, device='cuda', 
    return_targets=False, return_outputs=True, return_features=False
):
    # Pone la red en modo evaluación 
    model.eval()
    
    # Inicializa listas si son requeridas
    all_targets = [] if return_targets else None
    all_outputs = [] if return_outputs else None
    all_features = [] if return_features else None

    with torch.no_grad():
        for batch in dataloader:
            # Soporta tanto (inputs, targets) como solo inputs
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                inputs, targets = batch
                inputs = inputs.to(device)
                if return_targets:
                    all_targets.append(targets.cpu())
            else:
                inputs = batch
                inputs = inputs.to(device)

            # Modelado según los flags
            if return_features and return_outputs:
                features, outputs = model(inputs, mode='both')
                all_features.append(features.cpu())
                all_outputs.append(outputs.cpu())
            elif return_features:
                features = model(inputs, mode='features')
                all_features.append(features.cpu())
            elif return_outputs:
                outputs = model(inputs)
                all_outputs.append(outputs.cpu())

    # Concatena según sea necesario
    result = []
    if return_targets:
        result.append(torch.cat(all_targets))
    if return_features:
        result.append(torch.cat(all_features))
    if return_outputs:
        result.append(torch.cat(all_outputs))

    # Si solo hay un resultado, lo devuelve directamente
    return result[0] if len(result) == 1 else tuple(result)


def evaluate(model, dataloader, metric_fn=None, device='cuda', **kwargs):
    
    #
    all_targets, all_predicted = inference(model, dataloader, device)
    
    #
    metric_value = metric_fn(all_predicted, all_targets, **kwargs)

    return metric_value


#-------------------------------------------------------------------------------------------------------------
# 

# Establece el learning rate base y weight decay 
base_lr = 3e-2
wd = 2e-4

#-------------------------------------------------------------------------------------------------------------
# FINE-TUNING DE LA NUEVA CABECERA

# Congela los parámetros del extractor de características
for param in model.feature_extractor.parameters():
    param.requires_grad = False

# Configura el optimizador para el entrenamiento de la nueva cabecera (el módulo classifier)
optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=base_lr, weight_decay=wd)

# Numero de épocas que se entrena la nueva cabecera
NUM_EPOCHS_HEAD = 1

for epoch in range(NUM_EPOCHS_HEAD):

    # Entrena el modelo con el conjunto de entrenamiento
    head_train_loss = train(model, train_loader, criterion, optimizer, device=device)

    # Evalua el modelo con el conjunto de validación
    head_valid_loss = evaluate(model, valid_loader, criterion, device=device)

    # Imprime los valores de pérdida obtenidos en entrenamiento y validación 
    print(f"Epoch {epoch+1:>2} | Train Loss: {head_train_loss:>7.3f} | " + 
          f"Validation Loss: {head_valid_loss:>7.3f}")
    
# Guarda los pesos del modelo actual como los mejores hasta ahora
torch.save(model.state_dict(), args.model_path)

print("✅ Entrenamiento de la nueva cabecera completado\n")

#-------------------------------------------------------------------------------------------------------------
# DESCONGELA PARÁMETROS DEL MODELO Y ASIGNA LEARNING RATE DISCRIMINATIVO

# Descongela todos los parámetros del modelo
for param in model.parameters():
    param.requires_grad = True

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

# Genera una lista de tasas de aprendizaje para cada capa, aumentando de forma exponencial desde min_lr hasta 
# max_lr
lrs = [
    min_lr * (max_lr / min_lr) ** (i / (n_layers - 1)) 
    for i in range(n_layers)
]

# Lista en la que se almacenarán los parámetros por grupo y sus lr
param_groups = []
for layer_group, lr in zip(layer_groups, lrs):
    param_groups.append(
        {'params': layer_group, 'lr': lr}
    )

#-------------------------------------------------------------------------------------------------------------
# INICIALIZA LOS PARÁMETROS PARA EL EARLY STOPPING

# Número máximo de épocas a entrenar (si no se activa el early stopping)
MAX_EPOCHS = 30  

# Número mínimo de épocas a entrenar
MIN_EPOCHS = 30

# Número de épocas sin mejora antes de detener el entrenamiento
PATIENCE = 10

# Inicializa la mejor pérdida de validación como la obtenida en el entrenamiento de la cabecera
best_valid_loss = head_valid_loss   

# Contador de épocas sin mejora
epochs_no_improve = 0 

#-------------------------------------------------------------------------------------------------------------
# CONFIGURACIÓND EL OPTIMIZADOR Y SCHEDULER

# Configura el optimizador con los hiperparámetros escogidos
optimizer = torch.optim.AdamW(param_groups, lr=base_lr, weight_decay=wd)

# Crea el scheduler OneCycleLR
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, 
    max_lr=lrs, 
    #pct_start=MIN_EPOCHS/MAX_EPOCHS * 0.5,
    steps_per_epoch=len(train_loader),
    epochs=MAX_EPOCHS
)

#-------------------------------------------------------------------------------------------------------------
# ENTRENAMIENTO DE LA RED COMPLETA

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
    valid_loss = evaluate(model, valid_loader, criterion, device)
    valid_losses.append(valid_loss)
    
    # Imprime los valores de pérdida obtenidos en entrenamiento y validación  
    print(f"Epoch {epoch+1:>2} | Train Loss: {train_loss:>7.3f} | Validation Loss: {valid_loss:>7.3f}")
    
    # Comprueba si la pérdida en validación ha mejorado
    if valid_loss < best_valid_loss:
        
        # Actualiza la mejor pérdida en validación obtenida hasta ahora
        best_valid_loss = valid_loss
        
        # Reinicia el contador de épocas sin mejora si la pérdida ha mejorado
        epochs_no_improve = 0
        
        # Guarda los pesos del modelo actual como los mejores hasta ahora
        torch.save(model.state_dict(), args.model_path)
        
    else:
        # Incrementa el contador si no hay mejora en la pérdida de validación
        epochs_no_improve += 1

    # Si no hay mejora durante un número determinado de épocas (patience) y ya ha pasado el número mínimo de 
    # épocas, detiene el entrenamiento
    if epochs_no_improve >= PATIENCE and (epoch+1) > MIN_EPOCHS: 
        print(f"Early stopping at epoch {epoch+1}")
        break
    
    
# Carga los pesos del modelo que obtuvo la mejor validación
model.load_state_dict(torch.load(args.model_path))

# Cálculo de tiempo total de entrenamiento 
end_time = time.time()
elapsed_time = end_time - start_time

print("✅ Entrenamiento de la red completa completado\n")

#-------------------------------------------------------------------------------------------------------------
# ESTADÍSTICAS DE TIEMPO DEL ENTRENAMIENTO

# Convierte el tiempo en horas, minutos y segundos
hours = int(elapsed_time // 3600)
minutes = int((elapsed_time % 3600) // 60)
seconds = int(elapsed_time % 60)

# Imprime el tiempo de ejecución en formato horas:minutos:segundos
print(f"El entrenamiento y validación ha tardado {hours} horas, {minutes} minutos y {seconds} segundos.")

# Calcula el número total de épocas 
num_epochs = len(train_losses)

# Calcula el tiempo promedio por época
avg_elapsed_time_per_epoch = elapsed_time / num_epochs

# Convierte el tiempo promedio por época en minutos y segundos
minutes = int((avg_elapsed_time_per_epoch % 3600) // 60)
seconds = int(avg_elapsed_time_per_epoch % 60)

# Imprime el tiempo de ejecución medio de una época en formato minutos:segundos
print(f"En promedio, cada época del entrenamiento y la validación ha tardado {minutes} minutos y " + 
      f"{seconds} segundos.\n")

#-------------------------------------------------------------------------------------------------------------
# DIBUJADO Y GUARDADO DE CURVAS DE APRENDIZAJE

if hasattr(args, 'training_plot') and args.training_plot:
    
    # Grafica las curvas de aprendizaje
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(valid_losses, label='Validation Loss')
    
    # Calcular límites automáticos ignorando outliers
    upper_limit = train_losses[0]
    lower_limit = min(min(train_losses), min(valid_losses))
    plt.ylim(lower_limit * 0.95, upper_limit * 1.05) 
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid(True)

    # Guarda la imagen
    plt.savefig(args.training_plot, dpi=200, bbox_inches='tight')  

#-------------------------------------------------------------------------------------------------------------

if  args.pred_model_type=='CRF':
    
    # Obtiene valores verdaderos, características extraídas y valores predichos del conjunto de validación
    valid_true_values, valid_features, valid_pred_values = \
        inference(model, valid_loader, return_targets=True, return_features=True)
    
    valid_errors = torch.abs(valid_true_values - valid_pred_values)
    
    
    X = valid_features.numpy()
    y = valid_errors.numpy()
    
    # Dividir para entrenamiento y prueba del modelo auxiliar
    X_train_aux, X_test_aux, y_train_aux, y_test_aux = train_test_split(X, y, test_size=0.2, random_state=42)

    error_model = RandomForestRegressor(n_estimators=100, random_state=42)
    error_model.fit(X_train_aux, y_train_aux)

#-------------------------------------------------------------------------------------------------------------
# CALIBRACIÓN CONFORMAL

# Solo aplicamos calibración para modelos específicos (ICP y CQR)
if args.pred_model_type in ['ICP','CQR']:

    # Obtener predicciones y valores verdaderos del conjunto de calibración
    calib_pred_values, calib_true_values = inference(model, calib_loader)

    # Para ICP, calculamos las puntuaciones de calibración como la MAE
    if args.pred_model_type == 'ICP':
        
        # Calcula el nivel de cuantificación ajustado basado en el tamaño del conjunto de calibración y alpha
        n = len(calib_true_values)
        q_level = np.ceil((1-alpha) * (n + 1)) / n
        
        # Calcula las puntuaciones de calibración como valores absolutos de los errores
        calib_scores = np.abs(calib_true_values-calib_pred_values)
        
        # Calcula el cuantil qhat usado para ajustar el intervalo predictivo
        q_hat = np.quantile(calib_scores, q_level, method='higher')

    # Para CQR, calculamos las puntuaciones usando los límites inferior y superior de los intervalos predichos 
    elif args.pred_model_type == 'CQR':
        
        # Calcula el nivel de cuantificación ajustado basado en el tamaño del conjunto de calibración y alpha
        n = len(calib_true_values)
        q_level = np.ceil((1-alpha/2) * (n + 1)) / n

        # Calcula las puntuaciones para el límite inferior (diferencia entre predicción inferior y valor real)
        # y para el límite superior (diferencia entre valor real y predicción superior)
        calib_scores_lower_bound = calib_pred_values[:, 0] - calib_true_values
        calib_scores_upper_bound = calib_true_values - calib_pred_values[:,-1]
        
        # Calcula los cuantiles qhat para ambos límites del intervalo predictivo
        q_hat_lower = np.quantile(calib_scores_lower_bound, q_level, method='higher')
        q_hat_upper = np.quantile(calib_scores_upper_bound, q_level, method='higher')

    print("✅ Calibración completada\n")
    
#-------------------------------------------------------------------------------------------------------------
# GUARDADO DE MODELO ENTRENADO 

checkpoint = {
    'pred_model_type': args.pred_model_type,
    'model_state_dict': model.state_dict()
}

if args.pred_model_type in ('QR', 'ICP', 'CQR'):
    checkpoint['alpha'] = alpha

if args.pred_model_type == 'ICP':
    checkpoint['q_hat'] = q_hat

elif args.pred_model_type == 'CQR':
    checkpoint['q_hat_lower'] = q_hat_lower
    checkpoint['q_hat_upper'] = q_hat_upper


torch.save(checkpoint, args.model_path)

#-------------------------------------------------------------------------------------------------------------
# TEST

# Obtiene los valores predichos y los verdaderos 
test_true_values, test_pred_values = inference(model, test_loader)

# Determina las predicciones puntuales
if args.pred_model_type in ['base', 'ICP']:
    test_point_pred_values = test_pred_values
else:
    num_outputs = test_pred_values.shape[1]
    middle_idx = num_outputs // 2
    test_point_pred_values = test_pred_values[:, middle_idx] 

print("Métricas de las predicción puntual:")

# Calcula el MAE y lo imprime
test_mae = torch.mean(torch.abs(test_true_values - test_point_pred_values))
print(f"- Error Absoluto Medio (MAE) en test: {test_mae:.3f}")

# Calcula e imprime el MSE y lo imprime
test_mse = torch.mean((test_true_values - test_point_pred_values) ** 2)
print(f"- Error Cuadrático Medio (MSE) en test: {test_mse:.3f}")

# Si es un modelo con intervalos, calcula límites, cobertura y tamaño del intervalo
if args.pred_model_type in ['ICP', 'QR', 'CQR']:

    if args.pred_model_type == 'ICP':
        test_pred_lower_bound = test_point_pred_values - q_hat
        test_pred_upper_bound = test_point_pred_values + q_hat

    elif args.pred_model_type == 'QR':
        test_pred_lower_bound = test_pred_values[:, 0]
        test_pred_upper_bound = test_pred_values[:,-1]

    elif args.pred_model_type == 'CQR':
        test_pred_lower_bound = test_pred_values[:, 0] - q_hat_lower
        test_pred_upper_bound = test_pred_values[:,-1] + q_hat_upper
        
    # elif args.pred_model_type == 'CQR-d':
    

    print("Métricas de la predicción interválica:")

    # Calcula la cobertura empírica y lo imprime
    empirical_coverage = empirical_coverage(test_pred_lower_bound, test_pred_upper_bound, test_true_values)
    print(f"- Cobertura empírica (para {(1-alpha)*100}% de confianza): {empirical_coverage*100:>6.3f} %")

    # Calcula el tamaño medio del intervalo de predicción y lo imprime
    mean_interval_size = mean_interval_size(test_pred_lower_bound, test_pred_upper_bound)
    print(f"- Tamaño medio del intervalo: {mean_interval_size:>5.3f}")

print("✅ Testeo de la red completado\n")
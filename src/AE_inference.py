# ////////////////////////////////////////////////////////////////////////////////////////////////////////////
# //// PROBLEMA DE ESTIMACIÓN DE EDAD CON RADIOGRAFÍA MAXILOFACIAL
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
from sklearn.metrics import r2_score

# Manejo de tiempo y fechas
import time

# Modelos y funciones de pérdida personalizados 
from custom_models import ResNeXtRegressor, QuantileLoss
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


# Argumento 1: Ruta del modelo entrenado
parser.add_argument(
    '-m', '--model_path',
    type=str, 
    required=True,
    help="Archivo de destino para guardar el modelo entrenado (formato .pth)"
)

# Argumento 2: Ruta a las imágenes de entrada
parser.add_argument(
    '-i', '--input_path', 
    type=str, 
    required=True
)

# Argumento 3: Salida de resultados
parser.add_argument(
    '-o', '--output_stream',
    type=output_stream_type,
    default=sys.stdout,
    help="Archivo para redirigir TODA la salida (stdout + stderr). Usa '-' para terminal (por defecto)"
)

# Argumento 4: Append a la salida
parser.add_argument(
    '--append_output',
    action='store_true',
    help="Si se especifica, añade la salida al archivo en vez de sobrescribirlo."
)

# Argumento 5: Ignora los warnings
parser.add_argument('--ignore_warnings', action='store_true',
                    help='Ignora todos los warnings durante la ejecución')


# Parsear los argumentos
args = parser.parse_args()

# Validaciones adicionales
try:
    # Validación de extensión para el modelo
    validate_file_extension(args.model_path, ('.pth',), 'model_path')

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

print("✅ Argumentos parseados\n")

#-------------------------------------------------------------------------------------------------------------

transform = transforms.Compose(
    [transforms.Resize((448, 224)),
     transforms.ToTensor(),
     transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))]
)

class SingleImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        # En inferencia no tenemos targets, por lo que podemos devolver dummy target (por ejemplo 0)
        return image, torch.tensor(0)

# Comprobamos si es archivo o directorio
if os.path.isfile(args.input_path):
    image_paths = [args.input_path]
elif os.path.isdir(args.input_path):
    image_paths = [os.path.join(args.input_path, f) for f in os.listdir(args.input_path)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
else:
    raise ValueError("La ruta proporcionada no es válida")


# Creamos dataset y dataloader
dataset = SingleImageDataset(image_paths, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

#-------------------------------------------------------------------------------------------------------------
# CARGA DE MODELO

#
checkpoint = torch.load(args.model_path, weights_only=False)

#
pred_model_type = checkpoint['pred_model_type']
model_state_dict = checkpoint['model_state_dict']

#
if pred_model_type in ['base', 'splitCP']:
    
    # Inicializa el modelo de regresión estándar con una sola salida
    model = ResNeXtRegressor().to(device)
    
    # Carga los pesos del modelo 
    model.load_state_dict(model_state_dict)
    
elif pred_model_type in ['QR', 'CQR']:
    
    alpha = checkpoint['alpha']
    
    # Define los cuantiles que el modelo debe predecir 
    quantiles = [alpha/2, 0.5, 1-alpha/2]

    # Inicializa el modelo con múltiples salidas, una por cada cuantil
    model = ResNeXtRegressor(len(quantiles)).to(device)
    
    # Carga los pesos del modelo 
    model.load_state_dict(model_state_dict)


print("✅ Modelo cargado\n")

#-------------------------------------------------------------------------------------------------------------
# FUNCIÓN DE INFERENCIA 

def inference(model, dataloader, device='cuda'):
    
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

    # Concatena todas las predicciones y targets, y los devuelve
    all_predicted = torch.cat(all_predicted)
    all_targets = torch.cat(all_targets)

    return all_predicted, all_targets

#-------------------------------------------------------------------------------------------------------------

# Realiza la inferencia
pred_values, _ = inference(model, dataloader)

# Obtiene la predicción puntual
if pred_model_type in ['base','splitCP']:
    point_pred_values = pred_values
else:
    num_outputs = pred_values.shape[1]
    middle_idx = num_outputs // 2
    point_pred_values = pred_values[:, middle_idx] 
    
# Mustra las predicciones puntuales para cada imagen
for img_path, pred in zip(image_paths, point_pred_values):
    print(f"Predicción puntual para {os.path.basename(img_path)} con {pred_model_type}: {pred.item():.3f}")
    
#-------------------------------------------------------------------------------------------------------------

# Obtiene las predicciones interválicas 
if pred_model_type in ['splitCP', 'QR', 'CQR']:

    if pred_model_type == 'splitCP':
        test_pred_lower_bound = point_pred_values - checkpoint['q_hat']
        test_pred_upper_bound = point_pred_values + checkpoint['q_hat']

    elif pred_model_type == 'QR':
        pred_lower_bound = pred_values[:, 0]
        pred_upper_bound = pred_values[:,-1]

    elif pred_model_type == 'CQR':
        pred_lower_bound = pred_values[:, 0] - checkpoint['q_hat_lower']
        pred_upper_bound = pred_values[:,-1] + checkpoint['q_hat_upper']

    # Muestra las predicciones interválicas para cada imagen
    for img_path, lower, upper in zip(image_paths, pred_lower_bound, pred_upper_bound):
        print(f"Intervalo para {os.path.basename(img_path)} con {pred_model_type} "+
              f"(confianza del {(1-alpha)*100}%): [{lower.item():.3f}, {upper.item():.3f}]")

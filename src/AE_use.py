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

# Argumento 1: Tipo de modelo predictivo
# -------------------------------------------
# Selecciona el algoritmo de predicción entre las opciones implementadas
pred_model_types = ['base', 'QR', 'splitCP', 'CQR', 'MC-CQR']
parser.add_argument(
    '-t', '--pred_model_type',  # Cambiado de prediction_type para mayor precisión
    type=str,
    required=True,
    choices=pred_model_types,
    help='Tipo de modelo predictivo a utilizar. Opciones: ' + ', '.join(pred_model_types)
)

# Argumento 2: Nivel de confianza
# -----------------------------------------
# Valida que el nivel de confianza sea un float válido [0, 1]
def validate_confidence(x):
    x = float(x)
    if not 0.0 <= x <= 1.0:
        raise argparse.ArgumentTypeError(f"El nivel de confianza debe estar entre 0 y 1 (se recibió {x})")
    return round(x, 2) 

parser.add_argument(
    '-c', '--confidence_level', 
    type=validate_confidence,
    default=0.9,
    help='Nivel de confianza para intervalos de predicción (0.0 a 1.0). Valor por defecto: 0.9'
)

# Argumento 3: Modelo preentrenado
# --------------------------------------
# Ruta de un modelo existente para transfer learning o fine-tuning
parser.add_argument(
    '-l', '--load_model',  
    type=str,  
    required=True,
    help='Archivo con modelo preentrenado para inicialización (.pt)'
)

# Argumento 4: Salida de resultados
# --------------------------------------
# Controla dónde se escriben los resultados del proceso
parser.add_argument(
    '-o', '--output_stream',
    type=argparse.FileType('w', encoding='utf-8'),
    default=sys.stdout,
    help='Archivo para redirigir TODA la salida (stdout + stderr). Por defecto: terminal'
)

# Argumento 5: 
# --------------------------------------
# 
parser.add_argument(
    '-i', '--inference_image',
    type=argparse.FileType('r', encoding='utf-8'),
    default=None,
    help=''
)

# Argumento 6: Ignora los warnings
# --------------------------------------
parser.add_argument('--ignore_warnings', action='store_true',
                    help='Ignora todos los warnings durante la ejecución')


# Parsear los argumentos
args = parser.parse_args()

#
if not os.path.isfile(args.load_model):
    raise FileNotFoundError(f"No se encontró el archivo del modelo: {args.load_model}")

if not args.load_model.lower().endswith('.pth'):
    raise ValueError(f"El archivo para cargar los pesos del modelo debe ser .pth")

# Redirige salida estándar (stdout) y errores (stderr) al archivo especificado en --output_stream
if args.output_stream is not sys.stdout:
    sys.stdout = args.output_stream
    sys.stderr = args.output_stream

# Si 
if hasattr(args, 'ignore_warnings') and args.ignore_warnings:
    warnings.filterwarnings("ignore")

print("✅ Argumentos parseados\n")

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

# División para los modelos 'SplitCP' y 'CQR'
else: 
    
    # Divide el conjunto de datos completo de entrenamiento en tres subconjuntos de forma estratificada:
    # - Entrenamiento (72% de las instancias)    (68%)
    # - Validación (18% de las instancias)       (17%)
    # - Calibración (10% de las instancias)      (15%)

    train_idx, calib_idx = train_test_split(
        range(len(trainset)), train_size=0.85, shuffle=True, stratify=intAges
    )

    statify_aix = [intAges[i] for i in train_idx]
    train_idx, valid_idx = train_test_split(
        train_idx, train_size=0.8, shuffle=True, 
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
if args.pred_model_type in ['splitCP','QR','CQR', 'MC-CQR']:
    alpha = (1-args.confidence_level)

# 
if args.pred_model_type in ['base','splitCP']:

    # Inicializa el modelo de regresión estándar con una sola salida
    model = ResNeXtRegressor().to(device)
    
    # Carga los pesos del modelo que obtuvo la mejor validación
    model.load_state_dict(torch.load(args.load_model))

# ... para 'QR' y 'CQR'
else:
    
    # Define los cuantiles que el modelo debe predecir (p.ej., 0.05 y 0.95 para 90% de confianza)
    quantiles = [alpha/2, 0.5, 1-alpha/2]

    # Inicializa el modelo con múltiples salidas, una por cada cuantil
    model = ResNeXtRegressor(len(quantiles)).to(device)
    
    # Carga los pesos del modelo que obtuvo la mejor validación
    model.load_state_dict(torch.load(args.load_model))


print("✅ Modelo cargado\n")

#-------------------------------------------------------------------------------------------------------------
# FUNCIONES DE INFERENCIA y EVALUACIÓN 

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


def enable_dropout(model, p=None):
    """
    Activa todos los dropout layers durante la inferencia.
    Si se pasa p, modifica la probabilidad de dropout.
    """
    for m in model.modules():
        if isinstance(m, nn.Dropout):  # Más robusto que comparar el nombre de clase
            m.train()
            if p is not None:
                m.p = p  # Cambia la probabilidad de dropout


def mc_dropout_inference(model, dataloader, patience=5, min_delta=0.01, max_mc=100, p=0.5, device='cuda'):
    """
    Realiza inferencia con MC Dropout dinámico, deteniéndose cuando la varianza converge.

    Parámetros:
    - model: el modelo de PyTorch
    - dataloader: el dataloader con los datos de prueba
    - patience: número de iteraciones consecutivas en las que la varianza debe ser estable
    - min_delta: umbral mínimo de cambio en la varianza para considerarla estable
    - max_mc: número máximo de muestras MC por ejemplo
    - p: probabilidad de dropout (si queremos modificarla en inferencia)
    - device: 'cuda' o 'cpu'.
    """
    
    model.eval()
    enable_dropout(model, p)
    
    all_predicted = []
    all_targets = []
    
    # No calculamos gradientes (más rápido y consume menos memoria)
    with torch.no_grad():
        for inputs, targets in dataloader:
            
            # Obtiene las imágenes y sus valores objetivo
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 
            batch_size = inputs.shape[0]
            
            # Almacena las predicciones MC aquí
            preds = []
            
            # Contadores de paciencia por cada instancia del batch
            current_patience_count = torch.zeros(batch_size, dtype=torch.int, device=device)
            
            # Almacena la varianza de la iteración anterior para comparar
            prev_variance = None
            
            #
            for mc_iter in range(max_mc):
                
                # Hace forward pass con dropout activado
                outputs = model(inputs) # [total_batch_size, output_dim]
                
                # Añade una dimensión para acumular fácilmente
                preds.append(outputs.unsqueeze(0))
                
                # Convertimos la lista de predicciones a tensor de shape [mc_iter+1, batch_size, output_dim]
                preds_tensor = torch.cat(preds, dim=0)
                
                #  Calculamos la varianza de las predicciones actuales a lo largo de la dimensión MC
                variance = preds_tensor.std(dim=0) # Resultado: [batch_size, output_dim]
                
                if mc_iter > 0:
                    # Si ya tenemos al menos dos muestras, calculamos la diferencia de varianza
                    var_diff = torch.abs(variance - prev_variance)  # [batch_size, output_dim]
                    
                    # Comprobamos, para cada muestra del batch, si todas sus salidas están dentro de min_delta
                    stable = (var_diff <= min_delta).all(dim=1)  # Resultado: [batch_size], True si estable
                    
                    # Si es estable, aumenta la paciencia; si no, se resetea a 0
                    current_patience_count += stable.int()
                    current_patience_count *= stable.int()      
                
                # Guarda la varianza actual para comparar en la próxima iteración
                prev_variance = variance
                
                # Si todos los ejemplos del batch han superado el patience, para el bucle
                if (current_patience_count > patience).all():
                    break
            
            # Cuando termina el bucle (por paciencia o por max_mc), calcula la media MC
            preds_tensor = preds_tensor.mean(dim=0)
            
            # Guarda predicciones y targets para este batch
            all_predicted.append(preds_tensor.cpu())
            all_targets.append(targets.cpu())
            
    # Une todos los batches en un único tensor
    all_predicted = torch.cat(all_predicted, dim=0) # [total_batch_size, output_dim]
    all_targets = torch.cat(all_targets)
        
    return all_predicted, all_targets

#-------------------------------------------------------------------------------------------------------------
# CALIBRACIÓN

# Solo aplicamos calibración para modelos específicos (splitCP y CQR)
if args.pred_model_type in ['splitCP','CQR']:

    # Obtener predicciones y valores verdaderos del conjunto de calibración
    calib_pred_values, calib_true_values = inference(model, calib_loader)

    # Para splitCP, calculamos las puntuaciones de calibración como la MAE
    if args.pred_model_type == 'splitCP':
        
        # Calcula el nivel de cuantificación ajustado basado en el tamaño del conjunto de calibración y alpha
        n = len(calib_true_values)
        q_level = np.ceil((1-alpha) * (n + 1)) / n
        
        #
        calib_scores = np.abs(calib_true_values-calib_pred_values)
        
        # Calcula el cuantil qhat usado para ajustar el intervalo predictivo
        q_hat = np.quantile(calib_scores, q_level, method='higher')

    # Para CQR, calculamos las puntuaciones usando los límites inferior y superior de los intervalos predichos 
    elif args.pred_model_type == 'CQR':
        
        # Calcula el nivel de cuantificación ajustado basado en el tamaño del conjunto de calibración y alpha
        n = len(calib_true_values)
        q_level = np.ceil((1-alpha/2) * (n + 1)) / n

        #
        calib_scores_lower_bound = calib_pred_values[:, 0] - calib_true_values
        calib_scores_upper_bound = calib_true_values - calib_pred_values[:,-1]
        
        #
        q_hat_lower = np.quantile(calib_scores_lower_bound, q_level, method='higher')
        q_hat_upper = np.quantile(calib_scores_upper_bound, q_level, method='higher')

    print("✅ Calibración completada\n")


if args.pred_model_type=='MC-CQR':
    
    # Obtener predicciones y valores verdaderos del conjunto de calibración
    calib_pred_values, calib_true_values = mc_dropout_inference(model, calib_loader)
    
    # Calcula el nivel de cuantificación ajustado basado en el tamaño del conjunto de calibración y alpha
    n = len(calib_true_values)
    q_level = np.ceil((1-alpha/2) * (n + 1)) / n 
    
    #
    calib_scores_lower_bound = calib_pred_values[:, 0] - calib_true_values
    calib_scores_upper_bound = calib_true_values - calib_pred_values[:,-1]
    
    #
    q_hat_lower = np.quantile(calib_scores_lower_bound, q_level, method='higher')
    q_hat_upper = np.quantile(calib_scores_upper_bound, q_level, method='higher')
    
    print("✅ Calibración completada\n")

#-------------------------------------------------------------------------------------------------------------
# TEST

# Obtiene los valores predichos y los verdaderos 
test_pred_values, test_true_values = inference(model, test_loader) if args.pred_model_type!='MC-CQR' else mc_dropout_inference(model, test_loader)

# Determina las predicciones puntuales
if args.pred_model_type == 'base':
    test_point_pred_values = test_pred_values
else:
    num_outputs = test_pred_values.shape[1]
    middle_idx = num_outputs // 2
    test_point_pred_values = test_pred_values[:, middle_idx] 

# Calcula el MAE y lo imprime
test_mae = torch.mean(torch.abs(test_true_values - test_point_pred_values))
print(f"Error Absoluto Medio (MAE) en test: {test_mae:.3f}")

# Calcula e imprime el MSE y lo imprime
test_mse = torch.mean((test_true_values - test_point_pred_values) ** 2)
print(f"Error Cuadrático Medio (MSE) en test: {test_mse:.3f}")

# Si es un modelo con intervalos, calcula límites, cobertura y tamaño del intervalo
if args.pred_model_type in ['splitCP', 'QR', 'CQR', 'MC-CQR']:

    if args.pred_model_type == 'splitCP':
        test_pred_lower_bound = test_pred_values - q_hat
        test_pred_upper_bound = test_pred_values + q_hat

    elif args.pred_model_type == 'QR':
        test_pred_lower_bound = test_pred_values[:, 0]
        test_pred_upper_bound = test_pred_values[:,-1]

    elif args.pred_model_type in ['CQR', 'MC-CQR']:
        test_pred_lower_bound = test_pred_values[:, 0] - q_hat_lower
        test_pred_upper_bound = test_pred_values[:,-1] + q_hat_upper

    # Calcula la cobertura empírica y lo imprime
    empirical_coverage = empirical_coverage(test_pred_lower_bound, test_pred_upper_bound, test_true_values)
    print(f"Cobertura empírica (para {(1-alpha)*100}% de confianza): {empirical_coverage*100:>6.3f} %")

    # Calcula el tamaño medio del intervalo de predicción y lo imprime
    mean_interval_size = mean_interval_size(test_pred_lower_bound, test_pred_upper_bound)
    print(f"Tamaño medio del intervalo: {mean_interval_size:>5.3f}")


print("✅ Testeo de la red completado\n")


    
    


print("----------------------------------------------------------------------------------------------\n")



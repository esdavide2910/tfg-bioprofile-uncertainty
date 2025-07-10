# -*- coding: utf-8 -*-

# ////////////////////////////////////////////////////////////////////////////////////////////////////////////
# //// PROBLEMA DE ESTIMACIÓN DE EDAD CON RADIOGRAFÍA MAXILOFACIAL
# ////////////////////////////////////////////////////////////////////////////////////////////////////////////

# Biblioteca para aprendizaje profundo
import torch

# 
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms

#
if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA no está disponible. PyTorch no reconoce la GPU."
    )
device = 'cuda'

import traceback
import time

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
import polars as pl 

# Manejo y edición de imágenes
from PIL import Image

# Visualización de datos
import matplotlib.pyplot as plt
import seaborn as sns

# Operaciones aleatorias
import random

# Evaluación y partición de modelos
from sklearn.model_selection import train_test_split

# Modelos y funciones de pérdida personalizados 
from conformal_regression_models import *
from cp_metrics import *

#-------------------------------------------------------------------------------------------------------------
# CONFIGURACIÓN DE ARGUMENTOS PARA ENTRENAMIENTO Y EVALUACIÓN DE LOS MODELOS 

# Funciones auxiliares
def validate_confidence(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{x}' no es un número válido para el nivel de confianza")
    if not 0.0 <= x <= 1.0:
        raise argparse.ArgumentTypeError(f"El nivel de confianza debe estar entre 0 y 1 (se recibió {x})")
    return round(x, 2)

def validate_file_extension(filename, extensions, arg_name):
    if not any(filename.lower().endswith(ext) for ext in extensions):
        raise argparse.ArgumentTypeError(
            f"El archivo para '{arg_name}' debe tener una de las siguientes extensiones: "+
            f"{', '.join(extensions)}"
        )
    return filename

# Tipos de modelos de predicción soportados
PRED_MODEL_TYPES = ['base', 'ICP', 'QR', 'CQR']

# Agregado de argumentos 
def add_model_args(parser):
    parser.add_argument('--load_model_path', type=str)
    parser.add_argument('--save_model_path', type=str)
    parser.add_argument('--pred_model_type', type=str, choices=PRED_MODEL_TYPES)
    parser.add_argument('--confidence', type=validate_confidence, default=0.9)
    return parser

def add_training_args(parser):
    parser.add_argument('--train_head', action='store_true', 
                        help="Entrena solo la cabeza del modelo")
    parser.add_argument('--train', action='store_true', 
                        help="Entrena todo el modelo")
    parser.add_argument('--training_plot', type=str, default=None, 
                        help="Archivo para guardar el gráfico de entrenamiento (.png, .jpg, .jpeg)")
    parser.add_argument('--calibrate', action='store_true')
    return parser

def add_testing_args(parser):
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--save_test_results', type=str, default=None, 
                        help="Archivo para guardar los resultados de test en CSV")
    parser.add_argument('-i', '--test_iteration', type=int, default=None)
    return parser

def add_runtime_args(parser):
    parser.add_argument('--inference', type=str)
    parser.add_argument('--seed', type=int, default=22)
    parser.add_argument('--ignore_warnings', action='store_true', 
                        help="Ignora warnings")
    return parser

def add_output_args(parser):
    parser.add_argument('-o', '--output_stream', type=str, default=None, 
                        help="Archivo para redirigir salida (stdout + stderr)")
    parser.add_argument('--append_output', action='store_true', 
                        help="Añade en vez de sobrescribir salida")
    return parser

# Crea el parser
parser = argparse.ArgumentParser(
    description="Script para entrenamiento, calibración y/o evaluación del modelo para estimación de "+
                "edad a partir de radiografías maxilofaciales"
)

# Agrega todas las configuraciones al parser
add_model_args(parser)
add_training_args(parser)
add_testing_args(parser)
add_runtime_args(parser)
add_output_args(parser)

# Parsea los argumentos
args = parser.parse_args()

# Validaciones adicionales
try:
    
    if (args.train or args.train_head) and not args.save_model_path:
        parser.error("Debe especificar '--save_model_path' al entrenar el modelo.")
    
    if args.save_model_path:
        validate_file_extension(args.save_model_path, ['.pth'], 'save_model_path')
    
    if args.load_model_path:
        validate_file_extension(args.load_model_path, ['.pth'], 'load_model_path')
        if not os.path.isfile(args.load_model_path):
            parser.error(f"No se encontró el archivo del modelo en: {args.load_model_path}")
    
    if args.training_plot:
        if not args.train:
            parser.error("Se especificó '--training_plot' pero no se activó '--train'. "+
                         "No se generará ningún gráfico sin entrenamiento.")
        validate_file_extension(args.training_plot.name, ['.png', '.jpg', '.jpeg'], 'training_plot')
    
    if args.save_test_results and not args.test:
        parser.error("Se especificó '--save_test_results' pero no se activó '--test'. " +
                     "No se pueden guardar resultados si no se realiza una prueba.")
    
    if args.test_iteration and not args.save_test_results:
        parser.error("Se especificó '--test_iteration' pero no '--save_test_results'. " +
                     "El número de iteración solo tiene sentido si se guardan los resultados.")
    
    if args.save_test_results:
        validate_file_extension(args.save_test_results, ['.csv'], 'save_test_results')
    
    # Redirección de salida si se especifica
    if args.output_stream is not None: 
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

# Determina la semilla
SEED = args.seed

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
    worker_seed = torch.initial_seed() % 2**SEED
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
     transforms.ColorJitter(brightness=0.2, contrast=0.2), 
     transforms.ToTensor(),
     transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))]
)

# Define las transformaciones para las imágenes de validación y test, que son iguales que para entrenamiento 
# pero sin regularización
test_transform = transforms.Compose(
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
        metadata = pd.read_csv(metadata_file)  
        self.images_dir = images_dir
        self.transform = transform
        
        # Preprocesar los campos una sola vez
        self.img_paths = metadata['ID'].apply(lambda id_: os.path.join(images_dir, id_)).tolist()
        
        # Clasificación binaria de sexo: 1 para 'F' y 0 para 'M'
        self.sexes = torch.tensor((metadata['Sex'] != 'M').astype(int).values, dtype=torch.long)
        
        # Edad en float
        self.ages = torch.tensor(metadata['Age'].values, dtype=torch.float32)
    
    
    def __len__(self):
        return len(self.img_paths)
    
    
    def __getitem__(self, idx):
        
        # Carga la imagen
        image = Image.open(self.img_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, self.ages[idx]

# ------------------------------------------------------------------------------------------------------------

# Crea el Dataset de entrenamiento con augmentations
trainset = MaxillofacialXRayDataset(
    metadata_file = data_dir + 'metadata_train.csv',
    images_dir = data_dir + 'train/',
    transform = train_transform
)

# Crea el Dataset de validación con solo resize y normalización 
validset = MaxillofacialXRayDataset(
    metadata_file = data_dir + 'metadata_train.csv',
    images_dir = data_dir + 'train/',
    transform = test_transform
)

# Crea el Dataset de calibración con solo resize y normalización 
calibset = MaxillofacialXRayDataset(
    metadata_file = data_dir + 'metadata_train.csv',
    images_dir = data_dir + 'train/',
    transform = test_transform
)

# Crea el Dataset de test con solo resize y normalización
testset = MaxillofacialXRayDataset(
    metadata_file = data_dir + 'metadata_test.csv',
    images_dir = data_dir + 'test/',
    transform = test_transform
) 

# ------------------------------------------------------------------------------------------------------------

# Establece un batch size de 32 
BATCH_SIZE = 32

# Función optimizada para crear DataLoaders
def create_loader(dataset, indices=None, shuffle=False, num_workers=1):
    subset = Subset(dataset, indices) if indices is not None else dataset
    return DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g
    )

# Obtiene las edades del trainset por tramos de 0.5 años
halfAges = (np.floor(trainset.ages.numpy() * 2) / 2).astype(np.float32)
# Hay una única instancia con edad 26, que el algoritmo de separación de entrenamiento y validación será 
# incapaz de dividir de forma estratificada. Para evitar el error, reasigna esa instancia al tramo 
# inmediatamente inferior
halfAges[halfAges == 26.0] = 25.5

# Obtiene el sexo en binario
sexes = trainset.sexes.numpy()

# Crear etiquetas  combinadas de estratificación (p.ej.: "18.0_M", "17.5_F")
stratify_labels = np.array([f"{age:.1f}_{sex}" for age, sex in zip(halfAges, sexes)])

# Determina el número de hilos disponibles 
num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))

# División para los modelos 'base' y 'QR'
if args.pred_model_type in ['base','QR']:
    
    # Divide el conjunto de datos completo de entrenamiento en dos subconjuntos de forma estratificada:
    # - Entrenamiento (80% de las instancias)
    # - Validación (20% de las instancias)
    
    train_idx, valid_idx =  train_test_split(range(len(trainset)), train_size=0.8, shuffle=True, 
                                             random_state=SEED, stratify=stratify_labels)

    train_loader = create_loader(trainset, train_idx, shuffle=True, num_workers=num_workers)
    valid_loader = create_loader(validset, valid_idx)

# División para los modelos 'ICP' y 'CQR'
else: 
    
    # Divide el conjunto de datos completo de entrenamiento en tres subconjuntos de forma estratificada:
    # - Entrenamiento (68% de las instancias)
    # - Validación (17% de las instancias)
    # - Calibración (15% de las instancias)
    
    train_idx, calib_idx = train_test_split(range(len(trainset)), train_size=0.85, shuffle=True, 
                                            random_state=SEED, stratify=stratify_labels)
    
    train_idx, valid_idx = train_test_split(train_idx, train_size=0.8, shuffle=True, random_state=SEED,
                                            stratify=[stratify_labels[i] for i in train_idx])
    
    train_loader = create_loader(trainset, train_idx, shuffle=True, num_workers=num_workers)
    valid_loader = create_loader(validset, valid_idx)
    calib_loader = create_loader(calibset, calib_idx)

# Crea DataLoader de test
test_loader = create_loader(testset)

print("✅ Datasets de imágenes cargados\n")

#-------------------------------------------------------------------------------------------------------------
# CARGA DE MODELO Y DEFINICIÓN DE FUNCIÓN DE PÉRDIDA

MODEL_CLASSES = {
    'base': ResNeXtRegressor,
    'QR': ResNeXtRegressor_QR,
    'ICP': ResNeXtRegressor_ICP,
    'CQR': ResNeXtRegressor_CQR,
    # 'CRF': ResNeXtRegressor_CRF,
    # 'MCCQR': ResNeXtRegressor_MCCQR
}

# Obtiene los argumentos de tipo de modelo y nivel de confianza desde la línea de comandos
pred_model_type = args.pred_model_type
confidence = args.confidence

# Verifica que el tipo de modelo esté entre los tipos permitidos
if pred_model_type not in PRED_MODEL_TYPES:
    raise ValueError(f"Tipo de predicción desconocida: {pred_model_type}")

# Obtiene la clase del modelo correspondiente al tipo especificado
model_class = MODEL_CLASSES.get(pred_model_type)

# Instancia el modelo con el nivel de confianza especificado y lo envía a la GPU
model = model_class(confidence=confidence).to(device)

# Si se especificó una ruta para cargar un modelo previamente entrenado
if args.load_model_path:
    
    # Carga el checkpoint desde el archivo especificado 
    checkpoint = torch.load(args.load_model_path, weights_only=False)
    
    # Carga el modelo
    model.load_checkpoint(checkpoint)

print("✅ Modelo cargado\n")

#-------------------------------------------------------------------------------------------------------------
# FINE-TUNING DE LA CABECERA

if args.train or args.train_head:

    # Congela los parámetros del extractor de características
    for param in model.feature_extractor.parameters():
        param.requires_grad = False

    # Establece el weight decay 
    wd = 2e-4

    # Configura el optimizador para el entrenamiento de la cabecera 
    parameters = [
        {'params': model.classifier.fc2.parameters(), 'lr': 2e-2},
        {'params': model.classifier.fc1.parameters(), 'lr': 1e-2},
    ]
    optimizer = torch.optim.AdamW(parameters, weight_decay=wd)

    # Numero de épocas que se entrena la cabecera
    NUM_EPOCHS_HEAD = 3
    
    # Inicializa la mejor pérdida de validación como la obtenida en el entrenamiento de la cabecera
    best_valid_loss = float('inf')

    for epoch in range(NUM_EPOCHS_HEAD):
        
        # Inicia el temporizador para esta época
        start_time = time.time()

        # Entrena el modelo con el conjunto de entrenamiento
        head_train_loss = model.train_epoch(train_loader, optimizer)

        # Evalua el modelo con el conjunto de validación
        head_valid_loss = model.evaluate(valid_loader)
        
        # Calcula el tiempo transcurrido
        epoch_time = time.time() - start_time
        
        # Alternativa para mostrar minutos y segundos
        minutes = int(epoch_time // 60)
        seconds = int(epoch_time % 60)
        time_str = f"{minutes}m {seconds}s"

        # Imprime los valores de pérdida obtenidos en entrenamiento y validación 
        print(
            f"Epoch {epoch+1:>2} | "+
            f"Train Loss: {head_train_loss:>7.3f} | " + 
            f"Validation Loss: {head_valid_loss:>7.3f} | " +
            f"Time: {time_str}"
        )
        
        # Comprueba si la pérdida en validación ha mejorado
        if head_valid_loss < best_valid_loss:
            
            # Actualiza la mejor pérdida en validación obtenida hasta ahora
            best_valid_loss = head_valid_loss
            
            # Guarda los pesos del modelo actual como los mejores hasta ahora
            model.save_checkpoint(args.save_model_path)
    
    # Carga los pesos del modelo que obtuvo la mejor validación
    checkpoint = torch.load(args.save_model_path)
    model.load_checkpoint(checkpoint)

    print("✅ Entrenamiento de la cabecera completado\n")

#-------------------------------------------------------------------------------------------------------------
# ENTRENAMIENTO DE LA RED COMPLETA

if args.train:

    # Descongela todos los parámetros del modelo
    for param in model.parameters():
        param.requires_grad = True
        
    # Establece el weight decay
    wd = 5e-4

    # Establece las reglas para el learning rate discriminativo   
    max_lr = 1.5e-2         # Learning rate más alto (capa más superficial)
    min_lr = max_lr/100     # Learning rate más bajo (capa más profunda) 
    
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

    # Número de épocas a entrenar 
    NUM_EPOCHS = 30

    # Inicializa la mejor pérdida de validación como la obtenida en el entrenamiento de la cabecera
    best_valid_loss = float('inf')
    
    # Configura el optimizador con los hiperparámetros escogidos
    optimizer = torch.optim.AdamW(param_groups, weight_decay=wd)

    # Crea el scheduler OneCycleLR
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=lrs, 
        steps_per_epoch=len(train_loader),
        epochs=NUM_EPOCHS
    )
    
    # Listas para almacenar las pérdidas de entrenamiento y validación
    train_losses = []
    valid_losses = []

    # Bucle de entrenamiento por épocas
    for epoch in range(NUM_EPOCHS):
        
        # Inicia el temporizador para esta época
        start_time = time.time()
        
        # Entrena el modelo con el conjunto de entrenamiento
        train_loss = model.train_epoch(train_loader, optimizer, scheduler)
        train_losses.append(train_loss)
        
        # Evalua el modelo con el conjunto de validación
        valid_loss = model.evaluate(valid_loader) 
        valid_losses.append(valid_loss)
        
        # Calcula el tiempo transcurrido
        epoch_time = time.time() - start_time
        
        # Alternativa para mostrar minutos y segundos
        minutes = int(epoch_time // 60)
        seconds = int(epoch_time % 60)
        time_str = f"{minutes}m {seconds}s"

        # Imprime los valores de pérdida obtenidos en entrenamiento y validación  
        print(
            f"Epoch {epoch+1:>2} | " +
            f"Train Loss: {train_loss:>7.3f} | " +
            f"Validation Loss: {valid_loss:>7.3f} | " +
            f"Time: {time_str}"
        )
        
        # Comprueba si la pérdida en validación ha mejorado
        if valid_loss < best_valid_loss:
            
            # Actualiza la mejor pérdida en validación obtenida hasta ahora
            best_valid_loss = valid_loss
            
            # Guarda los pesos del modelo actual como los mejores hasta ahora
            model.save_checkpoint(args.save_model_path)
    
    
    # Carga los pesos del modelo que obtuvo la mejor validación
    checkpoint = torch.load(args.save_model_path)
    model.load_checkpoint(checkpoint)

    print("✅ Entrenamiento de la red completa completado\n")

#-------------------------------------------------------------------------------------------------------------
# DIBUJADO Y GUARDADO DE CURVAS DE APRENDIZAJE

    if args.training_plot:
        
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
# CALIBRACIÓN CONFORMAL

if args.calibrate and pred_model_type not in ['base', 'QR']:
    
    if pred_model_type == 'ICP':
        model.calibrate(calib_loader)
        
    elif pred_model_type == 'CQR':
        model.calibrate(calib_loader)
        
    elif pred_model_type == 'CRF':
        model.calibrate(calib_loader, valid_loader)
        
    # Guarda los parámetros de calibración del modelo
    model.save_checkpoint(args.save_model_path)
    
    print("✅ Calibración completada\n")

#-------------------------------------------------------------------------------------------------------------
# TEST

if args.test:

    # Obtiene las predicciones puntuales e interválicas
    if pred_model_type == 'base':
        test_pred_point_values, test_pred_lower_bound, test_pred_upper_bound, test_true_values = \
            model.inference(test_loader, valid_loader)
    else:
        test_pred_point_values, test_pred_lower_bound, test_pred_upper_bound, test_true_values = \
            model.inference(test_loader)
    
    #
    print("Métricas de las predicciones puntuales:")

    # Calcula el MAE y lo imprime
    test_mae = torch.mean(torch.abs(test_true_values - test_pred_point_values))
    print(f"- Error Absoluto Medio (MAE) en test: {test_mae:.3f}")

    # Calcula e imprime el MSE y lo imprime
    test_mse = torch.mean((test_true_values - test_pred_point_values) ** 2)
    print(f"- Error Cuadrático Medio (MSE) en test: {test_mse:.3f}")

    print("Métricas de las predicciones interválicas:")

    # Calcula la cobertura empírica y lo imprime
    EC = empirical_coverage(test_pred_lower_bound, test_pred_upper_bound, test_true_values)
    print(f"- Cobertura empírica: {EC*100:>6.3f} %")

    # Calcula el tamaño medio del intervalo de predicción y lo imprime
    MIW = mean_interval_size(test_pred_lower_bound, test_pred_upper_bound)
    print(f"- Tamaño medio del intervalo: {MIW:>5.3f}")

    print("✅ Testeo de la red completado\n")
    
#-------------------------------------------------------------------------------------------------------------
    
    if args.save_test_results is not None:
        
        #
        n = len(test_pred_point_values)
        new_df = pl.DataFrame({
            "pred_model_type": [pred_model_type] * n,
            "confidence": np.array([confidence] * n, dtype=np.float32),
            "iteration": [args.test_iteration] * n,
            "pred_point_value": np.array(test_pred_point_values, dtype=np.float32),
            "pred_lower_bound": np.array(test_pred_lower_bound, dtype=np.float32),
            "pred_upper_bound": np.array(test_pred_upper_bound, dtype=np.float32),
            "true_value": np.array(test_true_values, dtype=np.float32),
        })
        
        # Si el archivo ya existe, cargarlo y filtrar duplicados
        if os.path.exists(args.save_test_results):
            existing_df = pl.read_csv(args.save_test_results)
            
            # Forzar el esquema correcto para que coincida con new_df
            existing_df = existing_df.cast({
                "pred_model_type": pl.Utf8,
                "confidence": pl.Float32,
                "iteration": pl.Int64,
                "pred_point_value": pl.Float32,
                "pred_lower_bound": pl.Float32,
                "pred_upper_bound": pl.Float32,
                "true_value": pl.Float32
            })

            # Filtrar todas las filas que NO tienen el mismo conjunto clave
            mask = ~((existing_df["pred_model_type"] == pred_model_type) &
                    (existing_df["confidence"] == confidence) &
                    (existing_df["iteration"] == args.test_iteration))

            filtered_df = existing_df.filter(mask)

            # Concatenar el nuevo dataframe
            final_df = pl.concat([filtered_df, new_df])
        else:
            final_df = new_df

        # Guardar sobrescribiendo el archivo
        final_df.write_csv(args.save_test_results, separator=",", include_header=True)

#-------------------------------------------------------------------------------------------------------------

if args.inference is not None:
    
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
    dataset = SingleImageDataset(image_paths, transform=test_transform)
    new_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Realiza la inferencia    
    if pred_model_type == 'base':
        # Predicciones puntuales e interválicas 
        new_pred_point_values, new_pred_lower_bound, new_pred_upper_bound, new_true_values = \
            model.inference(new_loader, valid_loader)
    else:
        # Predicciones puntuales e interválicas 
        new_pred_point_values, new_pred_lower_bound, new_pred_upper_bound, new_true_values = \
            model.inference(new_loader)
            
    # Mustra las predicciones puntuales para cada imagen
    for img_path, pred in zip(image_paths, new_pred_point_values):
        print(f"Predicción puntual para {os.path.basename(img_path)} con {pred_model_type}: {pred.item():.3f}")
        
    # Muestra las predicciones interválicas para cada imagen
    for img_path, lower, upper in zip(image_paths, new_pred_lower_bound, new_pred_upper_bound):
        print(f"Intervalo para {os.path.basename(img_path)} con {pred_model_type}: "+
              f"[{lower.item():.3f}, {upper.item():.3f}]")


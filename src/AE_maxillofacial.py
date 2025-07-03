# -*- coding: utf-8 -*-

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
from custom_models import *
from coverage_metrics import *

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


# ARGUMENTOS PARA ENTRENAMIENTO DEL MODELO
# Argumento 'train'
parser.add_argument(
    '--train',
    action = 'store_true'
)

# Argumento tipo de predicción
PRED_MODEL_TYPES = ['base', 'QR', 'ICP', 'CRF', 'CQR']
parser.add_argument(
    '--pred_model_type',
    type = str,
    choices = PRED_MODEL_TYPES,
)

# Argumento ruta al fichero de dibujado de curvas de aprendizaje de la red
parser.add_argument(
    '--training_plot',
    type = argparse.FileType('wb'),
    default=None,
    help = "Archivo para guardar gráfico de curva de aprendizaje (.png, .jpg, .jpeg)"
)

# ARGUMENTOS PARA CALIBRACIÓN DEL MODELO
# Argumento 'calibrate'
parser.add_argument(
    '--calibrate',
    action = 'store_true'
)

# ARGUMENTOS PARA TEST DEL MODELO
# Argumento 'test'
parser.add_argument(
    '--test',
    action = 'store_true'
)

parser.add_argument(
    '--inference', 
    type=str,
    default=None
)

# OTROS ARGUMENTOS 
# Argumento para determinar la ruta de donde se carga el modelo
parser.add_argument(
    '--load_model_path',
    type = str,
    required = False
)

# Argumento para determinar la ruta donde se guarda el modelo
parser.add_argument(
    '--save_model_path',
    type = str,
    required = True
)

# Argumento para determinar el nivel de confianza
parser.add_argument(
    '--confidence',
    type = validate_confidence,
    default = 0.9
)

# Argumento para determinar si se realiza embedding del metadato 'sex'
parser.add_argument(
    '--sex_embedding',
    action = 'store_true',
    help = "Si se espececifica, añade información de sexo a la entrada del modelo"
)

# Argumento para determinar la ruta del archivo para salida de resultados
parser.add_argument(
    '-o', '--output_stream',
    type = str,
    default = None,
    help = "Archivo para redirigir toda la salida (stdout + stderr). Usa '-' para terminal (por defecto)"
)

# Argumento para añadir la salida al archivo
parser.add_argument(
    '--append_output',
    action = 'store_true',
    help = "Si se especifica, añade la salida al archivo en vez de sobrescribirlo."
)

# Argumento para cambiar la semilla
parser.add_argument('--seed', type=str, default = 23)

# Argumento para ignorar los warnings
parser.add_argument(
    '--ignore_warnings', action = 'store_true', help = "Ignora todos los warnings durante la ejecución"
)

# Parsear los argumentos
args = parser.parse_args()

# Validaciones adicionales
try:
    # Validación de extensión para la ruta del fichero de guardado del modelo
    validate_file_extension(args.save_model_path, ('.pth',), 'save_model_path')
    
    # Validación de extensión para la ruta del fichero de carga del modelo
    if hasattr(args, 'load_model_path') and args.load_model_path:
        validate_file_extension(args.load_model_path, ('.pth',), 'load_model_path')    
    
    # Validación de extensión para el gráfico si se especifica
    if hasattr(args, 'training_plot') and args.training_plot:
        validate_file_extension(args.training_plot.name, ('.png', '.jpg', '.jpeg'), 'training_plot')
    
    # Apertura segura del stream de salida
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

# Fija la semilla para las operaciones aleatorias en Python puro
random.seed(args.seed)

# Fija la semilla para las operaciones aleatorias en NumPy
np.random.seed(args.seed)

# Fija la semilla para los generadores aleatorios de PyTorch en CPU
torch.manual_seed(args.seed)

# Fija la semilla para todos los dispositivos GPU (todas las CUDA devices)
torch.cuda.manual_seed_all(args.seed)

# Desactiva la autooptimización de algoritmos en cudnn, que puede introducir no determinismo
# torch.backends.cudnn.benchmark = False

# Fuerza a cudnn a usar operaciones determinísticas (más lento pero reproducible)
# torch.backends.cudnn.deterministic = True

# Obliga a Pytorch a usar algoritmos determinísticos cuando hay alternativa. Si no la hay, lanza un error.
# torch.use_deterministic_algorithms(True)

# Función auxiliar para asegurar que cada worker de DataLoader use una semilla basada en la global
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**args.seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
 
# Generador de números aleatorios para DataLoader
g = torch.Generator()
g.manual_seed(args.seed)

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
        metadata = pd.read_csv(metadata_file)  # Cargar los metadatos
        
        # 
        self.img_ids = metadata['ID'].tolist()
        
        #
        self.sexes = torch.from_numpy(np.where(metadata['Sex'] == 'M', 0, 1)).long()
        
        #
        self.ages = torch.from_numpy(metadata['Age'].astype('float32').values)
        
        #
        self.images_dir = images_dir
        self.transform = transform
    
    
    def __len__(self):
        return len(self.img_ids)
    
    
    def __getitem__(self, idx):
        
        # Obteniene la imagen
        img_path = os.path.join(self.images_dir, self.img_ids[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Obtiene el sexo
        sex = self.sexes[idx]
        
        # Obtiene la edad
        age = self.ages[idx]
        
        return image, sex, age

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
intAges = torch.floor(trainset.ages).int()
# Hay una única instancia con edad 26, que el algoritmo de separación de entrenamiento y validación será 
# incapaz de dividir de forma estratificada. Para evitar el error, reasigna esa instancia a la edad 
# inmediatamente inferior
intAges[intAges==26]=25

# Función auxiliar para crear un DataLoader a partir de un subconjunto del dataset
def create_loader(dataset, indices):
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1, 
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

MODEL_CLASSES = {
    'base': ResNeXtRegressor,
    'QR': ResNeXtRegressor_QR,
    'ICP': ResNeXtRegressor_ICP,
    'CQR': ResNeXtRegressor_CQR,
    'CRF': ResNeXtRegressor_CRF
}

if args.load_model_path:
    
    #
    checkpoint = torch.load(args.load_model_path, weights_only=False)
    pred_model_type = checkpoint['pred_model_type']
    use_metadata = checkpoint['use_metadata']
    
    #
    if pred_model_type != args.pred_model_type:
        raise ValueError(f"El modelo especificado para cargar {args.pred_model_type} no concuerda en "+
                         f"tipo de predicción con el especificado por argumento {pred_model_type}")
        
    if pred_model_type not in PRED_MODEL_TYPES:
        raise ValueError(f"Tipo de predicción desconocida: {pred_model_type}")
    
    # Obtiene la clase del modelo basado en el checkpoint
    model_class = MODEL_CLASSES.get(checkpoint['pred_model_type'])
    
    #
    if use_metadata != args.sex_embedding:
        raise ValueError(f"El modelo especificado no es compatible en sus entradas con el cargado")
    
    # Crea el modelo
    if use_metadata: 
        model = model_class(confidence=args.confidence, use_metadata=True, meta_input_size=1).to(device)
    else:
        model = model_class(confidence=args.confidence).to(device)

    # Carga el modelo
    model.load_checkpoint(checkpoint)

else:
    
    #
    pred_model_type = args.pred_model_type
    use_metadata = args.sex_embedding
    
    #
    model_class = MODEL_CLASSES.get(pred_model_type)
    
    # Crea el modelo 
    if use_metadata: 
        model = model_class(confidence=args.confidence, use_metadata=True, meta_input_size=1).to(device)
    else:
        model = model_class(confidence=args.confidence).to(device)


print("✅ Modelo cargado\n")

#-------------------------------------------------------------------------------------------------------------
# FINE-TUNING DE LA NUEVA CABECERA Y EL EMBEDDING

if args.train:

    # Establece el learning rate base y weight decay 
    # base_lr = 3e-2
    wd = 2e-4

    # Congela los parámetros del extractor de características
    for param in model.feature_extractor.parameters():
        param.requires_grad = False

    # Configura el optimizador para el entrenamiento de la nueva cabecera 
    if use_metadata:
        # Lista de grupos de parámetros con diferentes configuraciones
        parameters = [
            {'params': model.classifier.fc2.parameters(), 'lr': 3e-2},
            {'params': model.classifier.fc1.parameters(), 'lr': 2e-2},
            {'params': model.embedding.parameters(), 'lr': 2e-2}
        ]
        optimizer = torch.optim.AdamW(parameters, weight_decay=wd)
    else:
        parameters = [
            {'params': model.classifier.fc2.parameters(), 'lr': 3e-2},
            {'params': model.classifier.fc1.parameters(), 'lr': 2e-2},
        ]
        optimizer = torch.optim.AdamW(parameters, weight_decay=wd)

    # Numero de épocas que se entrena la nueva cabecera
    NUM_EPOCHS_HEAD = 1

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
    
    model.save_checkpoint(args.save_model_path)

    print("✅ Entrenamiento de la nueva cabecera completado\n")

#-------------------------------------------------------------------------------------------------------------
# ENTRENAMIENTO DE LA RED COMPLETA

if args.train:

    # Establece el learning rate base y weight decay 
    base_lr = 3e-2
    wd = 5e-4

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

    # Número máximo de épocas a entrenar (si no se activa el early stopping)
    MAX_EPOCHS = 1

    # Número mínimo de épocas a entrenar
    MIN_EPOCHS = 15

    # Número de épocas sin mejora antes de detener el entrenamiento
    PATIENCE = 10

    # Inicializa la mejor pérdida de validación como la obtenida en el entrenamiento de la cabecera
    best_valid_loss = float('inf')

    # Contador de épocas sin mejora
    epochs_no_improve = 0 

    # Configura el optimizador con los hiperparámetros escogidos
    optimizer = torch.optim.AdamW(param_groups, lr=base_lr, weight_decay=wd)

    # Crea el scheduler OneCycleLR
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=lrs, 
        steps_per_epoch=len(train_loader),
        epochs=30
    )

    # Listas para almacenar las pérdidas de entrenamiento y validación
    train_losses = []
    valid_losses = []

    # Bucle de entrenamiento por épocas
    for epoch in range(MAX_EPOCHS):
        
        # Inicia el temporizador para esta época
        start_time = time.time()
        
        # Entrena el modelo con el conjunto de entrenamiento
        train_loss = model.train_epoch(train_loader, optimizer, scheduler)
        train_losses.append(train_loss)
        
        # Evalua el modelo con el conjunto de validación
        valid_loss = model.evaluate(test_loader) #------------------------------------------------------------ CAMBIAR
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
        if valid_loss < best_valid_loss and (epoch+1) > MIN_EPOCHS:
            
            # Actualiza la mejor pérdida en validación obtenida hasta ahora
            best_valid_loss = valid_loss
            
            # Reinicia el contador de épocas sin mejora si la pérdida ha mejorado
            epochs_no_improve = 0
            
            # Guarda los pesos del modelo actual como los mejores hasta ahora
            model.save_checkpoint(args.save_model_path)
            
        else:
            # Incrementa el contador si no hay mejora en la pérdida de validación
            epochs_no_improve += 1

        # Si no hay mejora durante un número determinado de épocas (patience) y ya ha pasado el número mínimo de 
        # épocas, detiene el entrenamiento
        if epochs_no_improve >= PATIENCE and (epoch+1) > MIN_EPOCHS: 
            print(f"Early stopping at epoch {epoch+1}")
            break
    
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
    
    # Inicia el temporizador previo a la inferencia
    start_time = time.time()

    if pred_model_type == 'base':
        # Solo predicciones puntuales
        test_pred_point_values, test_true_values = model.inference(test_loader)
    else:
        # Predicciones puntuales e interválicas 
        test_pred_point_values, test_pred_lower_bound, test_pred_upper_bound, test_true_values = \
            model.inference(test_loader)
    
    # Calcula el tiempo transcurrido
    epoch_time = time.time() - start_time
    
    # Alternativa para mostrar minutos y segundos
    minutes = int(epoch_time // 60)
    seconds = int(epoch_time % 60)
    time_str = f"{minutes}m {seconds}s"

    #
    print(f"Tiempo de inferencia del conjunto test: {time_str}")
    
    #
    print("Métricas de las predicciones puntuales:")

    # Calcula el MAE y lo imprime
    test_mae = torch.mean(torch.abs(test_true_values - test_pred_point_values))
    print(f"- Error Absoluto Medio (MAE) en test: {test_mae:.3f}")

    # Calcula e imprime el MSE y lo imprime
    test_mse = torch.mean((test_true_values - test_pred_point_values) ** 2)
    print(f"- Error Cuadrático Medio (MSE) en test: {test_mse:.3f}")
    
    #
    if pred_model_type == 'base':
        test_pred_lower_bound = test_pred_point_values - 2 * test_mae
        test_pred_upper_bound = test_pred_point_values + 2 * test_mae

    #
    print("Métricas de las predicciones interválicas:")

    # Calcula la cobertura empírica y lo imprime
    EC = empirical_coverage(test_pred_lower_bound, test_pred_upper_bound, test_true_values)
    print(f"- Cobertura empírica: {EC*100:>6.3f} %")

    # Calcula el tamaño medio del intervalo de predicción y lo imprime
    MPIW = mean_interval_size(test_pred_lower_bound, test_pred_upper_bound)
    print(f"- Tamaño medio del intervalo: {MPIW:>5.3f}")
    
    # Calcula el tamaño mínimo del intervalo de predicción y lo imprime
    min_interval_size = quantile_interval_size(test_pred_lower_bound, test_pred_upper_bound, 0.0)
    print(f"- Tamaño mínimo de los intervalos: {min_interval_size:>5.3f}")
    
    # Calcula el tamaño máximo del intervalo de predicción y lo imprime
    max_interval_size = quantile_interval_size(test_pred_lower_bound, test_pred_upper_bound, 1.0)
    print(f"- Tamaño máximo de los intervalos: {max_interval_size:>5.3f}")

    print("✅ Testeo de la red completado\n")
    
    
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
        # Solo predicciones puntuales
        new_pred_point_values, new_true_values = model.inference(new_loader)
        new_pred_lower_bound = new_pred_point_values - 2 * test_mae
        new_pred_upper_bound = new_pred_point_values + 2 * test_mae
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
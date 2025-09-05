# -*- coding: utf-8 -*-

# ////////////////////////////////////////////////////////////////////////////////////////////////////////////
# //// PROBLEMA DE CLASIFICACIÓN DE MAYORÍA DE EDAD CON RADIOGRAFÍA MAXILOFACIAL
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

#-------------------------------------------------------------------------------------------------------------

import os
working_dir = os.getcwd()
data_dir = working_dir + '/data/AE_maxillofacial/preprocessed/'

#-------------------------------------------------------------------------------------------------------------

# Manejo del sistema y argumentos de línea de comandos
import sys
import argparse

# Control de errores y advertencias
import traceback
import warnings

# Medición de tiempo y pausas
import time

# Operaciones aleatorias
import random

# Manipulación de datos
import numpy as np
import pandas as pd

# Manejo y edición de imágenes
from PIL import Image

# Visualización de datos
import matplotlib.pyplot as plt

# Evaluación y partición de modelos
from sklearn.model_selection import train_test_split

# Modelos, funciones de pérdida y métricas personalizados 
from conformal_classification_models import *
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

#
SET_PRED_METHODS = ['base', 'LAC', 'MCM']
CONFORMAL_PRED_METHODS = ['LAC', 'MCM']

# Agregado de argumentos 
def add_model_args(parser):
    parser.add_argument('--load_model_path', type=str)
    parser.add_argument('--save_model_path', type=str)
    parser.add_argument('--pred_method', type=str, choices=SET_PRED_METHODS)
    parser.add_argument('--confidence', type=validate_confidence, default=0.95)
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
                "mayoría/minoría de edad a partir de radiografías maxilofaciales"
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

# Obtiene de los argumentos el tipo de predicción, nivel de confianza y número de iteración 
pred_method = args.pred_method
confidence = args.confidence
test_iteration = args.test_iteration

print("-----------------------------------------------\n")
print("Método: ", pred_method)
print("Nivel de confianza: ", confidence)
print("Iteración: ", test_iteration)

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
        
        # Sexo como valor string: 'M' para masculino y 'F' para femenino
        self.sexes = metadata['Sex']
        
        # Edad en float
        self.ages = torch.tensor(metadata['Age'].values, dtype=torch.float32)
        
        #
        self.__init_classes__()
        
    
    def __init_classes__(self):
        
        # Clasificación binaria de edad: 1 si >= 18, 0 si < 18
        self.labels = (self.ages >= 18).long()
        
        #
        self.num_classes = len(torch.unique(self.labels))
    
    
    def __len__(self):
        return len(self.img_paths)
    
    
    def __getitem__(self, idx):
        
        # Carga la imagen
        image = Image.open(self.img_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, self.labels[idx]


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
sexes = torch.tensor((trainset.sexes != 'M').astype(int).values, dtype=torch.long).numpy()

# Crear etiquetas  combinadas de estratificación (p.ej.: "18.0_M", "17.5_F")
stratify_labels = np.array([f"{age:.1f}_{sex}" for age, sex in zip(halfAges, sexes)])

# Determina el número de hilos disponibles 
num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))

# División para el modelo base
if args.pred_method == 'base':
    
    # Divide el conjunto de datos completo de entrenamiento en dos subconjuntos de forma estratificada:
    # - Entrenamiento (80% de las instancias)
    # - Validación (20% de las instancias)
    
    train_idx, valid_idx =  train_test_split(range(len(trainset)), train_size=0.8, shuffle=True, 
                                             random_state=SEED, stratify=stratify_labels)
    
    train_loader = create_loader(trainset, train_idx, shuffle=True, num_workers=num_workers)
    valid_loader = create_loader(validset, valid_idx)

# División para modelos CP
else: 
    
    # Divide el conjunto de datos completo de entrenamiento en tres subconjuntos de forma estratificada:
    # - Entrenamiento (68% de las instancias)
    # - Validación (17% de las instancias)
    # - Calibración (15% de las instancias)
    
    train_idx, calib_idx = train_test_split(range(len(trainset)), train_size=0.8, shuffle=True, 
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
    'base': ResNeXtClassifier,
    'LAC': ResNeXtClassifier_LAC,
    'MCM': ResNeXtClassifier_MCM
}

# Verifica que el tipo de modelo esté entre los tipos permitidos
if pred_method not in SET_PRED_METHODS:
    raise ValueError(f"Tipo de predicción desconocida: {pred_method}")

# Obtiene la clase del modelo correspondiente al tipo especificado
model_class = MODEL_CLASSES.get(pred_method)

# Instancia el modelo con el nivel de confianza especificado y lo envía a la GPU
model = model_class(num_classes=trainset.num_classes, confidence=confidence).to(device)

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
        {'params': model.classifier.fc2.parameters(), 'lr': 3e-2},
        {'params': model.classifier.fc1.parameters(), 'lr': 2e-2},
    ]
    optimizer = torch.optim.AdamW(parameters, weight_decay=wd)

    # Numero de épocas que se entrena la cabecera
    NUM_EPOCHS_HEAD = 5
    
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
            f"Train Loss: {head_train_loss:>6.3f} | " + 
            f"Validation Loss: {head_valid_loss:>6.3f} | " +
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

    # Configura el optimizador con los hiperparámetros escogidos
    optimizer = torch.optim.AdamW(param_groups, weight_decay=wd)

    # Crea el scheduler OneCycleLR
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=lrs,
        steps_per_epoch=len(train_loader),
        epochs=NUM_EPOCHS
    )
    
    # Inicializa la mejor pérdida de validación como la obtenida en el entrenamiento de la cabecera
    best_valid_loss = float('inf')
    
    # 
    best_epoch = -1

    # Bucle de entrenamiento por épocas
    for epoch in range(NUM_EPOCHS):
        
        # Inicia el temporizador para esta época
        start_time = time.time()
        
        # Entrena el modelo con el conjunto de entrenamiento
        train_loss = model.train_epoch(train_loader, optimizer, scheduler)
        
        # Evalua el modelo con el conjunto de validación
        valid_loss = model.evaluate(valid_loader) 
        
        # Calcula el tiempo transcurrido
        epoch_time = time.time() - start_time
        
        # Alternativa para mostrar minutos y segundos
        minutes = int(epoch_time // 60)
        seconds = int(epoch_time % 60)
        time_str = f"{minutes}m {seconds}s"

        # Imprime los valores de pérdida obtenidos en entrenamiento y validación  
        print(
            f"Epoch {epoch+1:>2} | " +
            f"Train Loss: {train_loss:>6.3f} | " +
            f"Validation Loss: {valid_loss:>6.3f} | " +
            f"Time: {time_str}"
        )
        
        # Comprueba si la pérdida en validación ha mejorado
        if valid_loss < best_valid_loss:
            
            #
            best_epoch = epoch + 1
            
            # Actualiza la mejor pérdida en validación obtenida hasta ahora
            best_valid_loss = valid_loss
            
            # Guarda los pesos del modelo actual como los mejores hasta ahora
            model.save_checkpoint(args.save_model_path)
    
    
    if valid_loss > best_valid_loss:
        
        #
        print(f"Restaurando los parámetros de la época {best_epoch}")
        
        # Carga los pesos del modelo que obtuvo la mejor validación
        checkpoint = torch.load(args.save_model_path)
        model.load_checkpoint(checkpoint)

    print("✅ Entrenamiento de la red completa completado\n")

#-------------------------------------------------------------------------------------------------------------
# CALIBRACIÓN DE PROBABILIDADES

if args.train or args.train_head:
    #
    model.set_temperature(valid_loader)

#-------------------------------------------------------------------------------------------------------------
# CALIBRACIÓN CONFORMAL

if args.calibrate and pred_method in CONFORMAL_PRED_METHODS:
    
    #
    model.calibrate(calib_loader)
        
    # Guarda los parámetros de calibración del modelo
    model.save_checkpoint(args.save_model_path)
    
    print("✅ Calibración completada\n")

#-------------------------------------------------------------------------------------------------------------
# TEST

if args.test:

    #
    test_pred_classes, test_pred_sets, test_true_classes = model.inference(test_loader)

    # Calcula y muestra la exactitud
    accry = (test_pred_classes == test_true_classes).sum() / test_true_classes.size(0)
    print(f"Accuracy: {accry*100:>4.2f} %")
    
    # Calcula y muestra la cobertura empírica 
    ec = empirical_coverage_classification(test_pred_sets, test_true_classes)
    print(f"Cobertura empírica: {ec*100:>4.2f} %")
    
    # Calcula y muestra el tamaño medio del conjunto
    mss = mean_set_size(test_pred_sets)
    print(f"Tamaño medio de conjunto: {mss:>4.2f}")
    
    # Calcula y muestra el ratio de indeterminación
    ir = indeterminancy_rate(test_pred_sets)
    print(f"Ratio de indeterminación: {ir*100:>4.2f} %")

    print("✅ Testeo de la red completado\n")
    
#-------------------------------------------------------------------------------------------------------------
    
    if args.save_test_results:
        
        # 
        n = len(test_pred_classes)
        new_df = pd.DataFrame({
            "pred_method": [pred_method] * n,
            "confidence": np.array([confidence] * n, dtype=np.float32),
            "iteration": [args.test_iteration] * n,
            "pred_class": np.array(test_pred_classes, dtype=np.uint8),
            "pred_set_under_18": np.array(test_pred_sets[:, 0], dtype=np.uint8),
            "pred_set_over_18":  np.array(test_pred_sets[:, 1], dtype=np.uint8),
            "true_class": np.array(test_true_classes, dtype=np.uint8),
            "age": testset.ages,
            "sex": testset.sexes
        })

        # Si el archivo ya existe, cargarlo y filtrar duplicados
        if os.path.exists(args.save_test_results):
            existing_df = pd.read_csv(args.save_test_results, dtype={
                "pred_method": str,
                "confidence": np.float32,
                "iteration": np.int64,
                "pred_class": np.uint8,
                "pred_set_under_18": np.uint8,
                "pred_set_over_18": np.uint8,
                "true_class": np.uint8,
                "age": np.float32,
                "sex": str
            })

            # Filtrar todas las filas que NO tienen el mismo conjunto clave
            mask = ~(
                (existing_df["pred_method"] == pred_method) &
                (existing_df["confidence"] == confidence) &
                (existing_df["iteration"] == args.test_iteration)
            )

            filtered_df = existing_df[mask]

            # Concatenar el nuevo dataframe
            final_df = pd.concat([filtered_df, new_df], ignore_index=True)
        else:
            final_df = new_df

        # Guardar sobrescribiendo el archivo
        final_df.to_csv(args.save_test_results, index=False, float_format="%.6f")
        

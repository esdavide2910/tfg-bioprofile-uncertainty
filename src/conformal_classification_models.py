#-------------------------------------------------------------------------------------------------------------
# BIBLIOTECAS ------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torchvision 
import math
from cp_metrics import mean_set_size
from typing import Tuple
from scipy.optimize import minimize_scalar

#-------------------------------------------------------------------------------------------------------------


#-------------------------------------------------------------------------------------------------------------
# COMPONENTES BÁSICOS ----------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------

class FeatureExtractorResNeXt(nn.Module):
    
    def __init__(self):
        
        super().__init__()
        
        # Carga el modelo con pesos pre-entrenados
        resnext = torchvision.models.resnext50_32x4d(weights='DEFAULT')
        
        self.conv1 = nn.Sequential(
            resnext.conv1,
            resnext.bn1,
            resnext.relu,
            resnext.maxpool,
        )
        self.conv2 = resnext.layer1
        self.conv3 = resnext.layer2
        self.conv4 = resnext.layer3
        self.conv5 = resnext.layer4
    
    
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        
        return x

#-------------------------------------------------------------------------------------------------------------

class ClassifierResNeXt(nn.Module):
    
    def __init__(self, input_size=2048, output_size=1):

        super(ClassifierResNeXt, self).__init__()  
        
        self.fc1 = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p = 0.5),
            nn.Linear(input_size, 512),
        )

        self.fc2 = nn.Sequential(
            nn.BatchNorm1d(512),
            nn.Dropout(p = 0.5),
            nn.Linear(512, output_size) 
        )
    
    
    def forward(self, x):
        
        x = self.fc1(x)
        x = self.fc2(x)
        
        # nn.Linear devuelve por defecto una salida 2D
        # Aplana la salida si solo hay una variable target
        outputs = x.squeeze(-1) if x.dim() > 1 and x.shape[-1] == 1 else x

        return outputs

#-------------------------------------------------------------------------------------------------------------


#-------------------------------------------------------------------------------------------------------------
# MODELOS COMPLETOS ------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------

class ResNeXtClassifier(nn.Module):
    
    def __init__(self, num_classes, **kwargs):
        
        # Inicializa la clase padre
        super().__init__()
        
        # Almacena el número de clases de clasificación
        self.num_classes = num_classes
        
        # Define las componentes de la red: 
        # 1) Extractor de características, pooling y flattening 
        self.feature_extractor = FeatureExtractorResNeXt()
        self.pool_avg = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()
        
        # 2) Classifier según el tipo de clasificación
        input_size = 2048 # características aplanadas 
        self.output_size = 1 if num_classes==2 else num_classes
        self.classifier = ClassifierResNeXt(input_size, self.output_size)
        
        # Define la función de pérdida
        self.loss_function = nn.BCEWithLogitsLoss() if num_classes==2 else nn.CrossEntropyLoss()
        
        # Define la temperatura ... 
        self.temperature = nn.Parameter(torch.ones(1) * 1.5) 
    
    
    def save_checkpoint(self, save_model_path):
        """Guarda el estado actual de un modelo y parámetros de calibración en un archivo"""
        checkpoint = {
            'pred_model_type': 'base',
            'num_classes': self.num_classes,
            'torch_state_dict': self.state_dict()
        }
        torch.save(checkpoint, save_model_path)
    
    
    def load_checkpoint(self, checkpoint):
        """Carga el estado del modelo desde un checkpoint"""
        # Extrae el número de salidas del modelo guardado
        weight_key = 'classifier.fc2.2.weight'
        bias_key = 'classifier.fc2.2.bias'
        
        if weight_key in checkpoint['torch_state_dict']:
            out_features = checkpoint['torch_state_dict'][weight_key].shape[0]
            
            if out_features != self.output_size:
                checkpoint['torch_state_dict'].pop(weight_key, None)
                checkpoint['torch_state_dict'].pop(bias_key, None)
        self.load_state_dict(checkpoint['torch_state_dict'], strict=False)
    
    
    def forward(self, image):
        """Paso forward del modelo"""
        
        # Extrae y aplana las características
        x = self.feature_extractor(image)
        x = self.pool_avg(x)
        x = self.flatten(x)
        
        # Pasa por el clasificador para obtener las predicciones
        logits =  self.classifier(x)
        
        # Ajusta la dimensión de salida si es necesario
        logits = logits.squeeze(-1) if logits.dim() > 1 and logits.shape[-1] == 1 else logits
        
        # Aplica temperature scaling solo en modo evaluación
        if not self.training:
            logits = self.temperature_scale(logits)
        
        return logits
    
    
    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        if self.num_classes == 2:
            # Logits de forma (batch,) o (batch, 1) — solo divide directamente
            return logits / self.temperature
        else:
            # Logits de forma (batch, num_classes)
            temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
            return logits / temperature
    
    
    def set_temperature(self, valid_loader):
        
        # Pone la red en modo evaluación 
        self.eval()
        
        # Listas para almacenar logits y etiquetas verdaderas de todo el conjunto de validación
        logits_list = []
        true_labels_list = []
        
        # Desactiva el cálculo de gradientes para eficiencia
        with torch.no_grad():
            for images, true_labels in valid_loader:
                images, true_labels = images.cuda(), true_labels.cuda()
                # Obtiene los logits del modelo sin aplicar temperature scaling
                logits = self.forward(images)  
                logits_list.append(logits)
                true_labels_list.append(true_labels)
        
        # Concatena todos los logits y etiquetas en un solo tensor para procesarlos juntos
        logits = torch.cat(logits_list)
        true_labels = torch.cat(true_labels_list)
        
        # Define el optimizador LBFGS para ajustar la temperatura (parámetro único)
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        
        # Función requerida por LBFGS para reevaluar la función objetivo y los gradientes
        def eval():
            optimizer.zero_grad()
            if self.num_classes == 2:
                loss = self.loss_function(self.temperature_scale(logits), true_labels.float())
            else:
                loss = self.loss_function(self.temperature_scale(logits), true_labels.long())
            loss.backward()
            return loss

        # Ejecuta la optimización para encontrar la temperatura que minimiza la pérdida
        optimizer.step(eval)
    
    
    def get_layer_groups(self):
        """Devuelve los parámetros del modelo agrupados por capas, de más superficiales a más profundas"""
        layer_groups = []
        layer_groups.append(list(self.feature_extractor.conv1.parameters()))
        layer_groups.append(list(self.feature_extractor.conv2.parameters()))
        layer_groups.append(list(self.feature_extractor.conv3.parameters()))
        layer_groups.append(list(self.feature_extractor.conv4.parameters()))
        layer_groups.append(list(self.feature_extractor.conv5.parameters()))
        layer_groups.append(list(self.classifier.fc1.parameters()))
        layer_groups.append(list(self.classifier.fc2.parameters()))
        return layer_groups
    
    
    def train_epoch(self, dataloader, optimizer, scheduler=None, loss_fn=None):
        """Entrena el modelo por una época completa"""
        
        # Determinamos la función de pérdida
        loss_fn = loss_fn if loss_fn is not None else self.loss_function 
        
        # Pone la red en modo entrenamiento 
        self.train()
        
        # Inicializa la pérdida acumulada para esta época
        epoch_loss = 0
        
        # Itera sobre todos los batches del dataloader
        for images, labels in dataloader:
            
            # Obtiene las imágenes de entrenamiento y sus clases verdaderas
            images, labels = images.to('cuda'), labels.to('cuda')
            
            # Limpia los gradientes de la iteración anterior
            optimizer.zero_grad()
            
            # Obtiene las predicciones del modelo
            logits = self.forward(images)
            
            # Calcula la pérdida de las predicciones
            if self.num_classes == 2:
                loss = loss_fn(logits, labels.float())
            else:
                loss = loss_fn(logits, labels.long())
            
            # Realiza la retropropagación para calcular los gradientes (propagación hacia atrás)
            loss.backward()
            
            # Actualiza los parámetros del modelo
            optimizer.step()
            
            # Actualiza el scheduler de la tasa de aprendizaje (si se proporciona)
            if scheduler is not None:
                scheduler.step()
            
            # Acumula la pérdida de este batch
            epoch_loss += loss.item()  
        
        # Calcula la pérdida promedio de la época y la devuelve
        avg_loss = epoch_loss / len(dataloader)
        return avg_loss
    
    
    def _inference(self, dataloader, return_probs=False):
        
        # Pone la red en modo evaluación 
        self.eval()
        
        # Inicializa listas si son requeridas
        all_outputs = []
        all_true_labels = [] 
        
        # Flag para detectar si el dataloader entrega etiquetas
        has_true_labels = False
        
        # Desactiva el cálculo de gradientes para eficiencia
        with torch.no_grad():
            for batch in dataloader:
                
                # Verifica si el batch contiene (image, target) o solo (image)
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    images, true_labels = batch 
                    images = images.to('cuda')
                    has_true_labels = True
                    all_true_labels.append(true_labels.cpu())
                else:
                    images = batch
                    images = images.to('cuda')
                
                # Forward pass
                logits = self.forward(images)
                
                if return_probs:
                    if self.num_classes == 2:
                        # Clasificación binaria: usa sigmoide y empaqueta como (n, 2)
                        p = torch.sigmoid(logits)
                        pred_scores = torch.stack([1 - p, p], dim=1).squeeze()
                    else:
                        # Clasificación multiclase: usa softmax
                        pred_scores = torch.softmax(logits, dim=1)
                    
                    all_outputs.append(pred_scores.cpu())
                else:
                    all_outputs.append(logits.cpu())
        
        # Concatena los resultados
        outputs = torch.cat(all_outputs)
        true_labels = torch.cat(all_true_labels) if has_true_labels else None
        
        # Devuelve ... 
        return outputs, true_labels
    
    
    def inference(self, dataloader):
        
        # Obtiene las probabilidades predichas para cada clase y la etiqueta verdadera para cada instancia
        pred_scores, true_labels = self._inference(dataloader, return_probs=True)

        # Determina la clase predicha (la de mayor probabilidad) para cada instancia
        _, pred_classes = torch.max(pred_scores, dim=1)
        
        # Convierte las clases predichas a one-hot encoding
        pred_sets = nn.functional.one_hot(pred_classes, num_classes=self.num_classes)
        
        return pred_classes, pred_sets, true_labels
    
    
    # def evaluate(self, dataloader, metric_fn=None):
    #     """Evalúa el modelo en un conjunto de datos"""
        
    #     # Determina la función de métrica
    #     metric_fn = metric_fn if metric_fn is not None else self.loss_function 
        
    #     # Obtiene las probabilidades predichas para cada clase y la clase verdadera para cada instancia
    #     pred_scores, true_classes = self._inference(dataloader, return_probs=True)
        
    #     # Determina la clase  predicha como la clase  con mayor puntuación
    #     _, pred_classes = torch.max(pred_scores, dim=1) 
        
    #     # Calcula el valor de la métrica y lo devuelve
    #     return metric_fn(pred_classes, true_classes)
    
    
    def evaluate(self, dataloader):
        """Evalúa el modelo en un conjunto de datos"""
        
        # Determinamos la función de pérdida
        loss_fn = self.loss_function 
        
        # Obtiene las probabilidades predichas para cada clase y la clase verdadera para cada instancia
        logits, true_labels = self._inference(dataloader)
        
        # Calcula la pérdida en todos los datos
        if self.num_classes == 2:
            loss = loss_fn(logits, true_labels.float()).item()
        else:
            loss = loss_fn(logits, true_labels.long()).item()
        
        return loss

#-------------------------------------------------------------------------------------------------------------

class ResNeXtClassifier_LAC(ResNeXtClassifier):
    
    PRED_MODEL_TYPE = 'LAC'
    
    def __init__(self, num_classes, confidence=0.9):
        
        # Inicializa la clase padre con los parámetros base
        super().__init__(num_classes)
        
        # Parámetros para la conformal prediction
        self.alpha = 1-confidence
        self.q_hat = None 
    
    
    def save_checkpoint(self, save_model_path):
        """Guarda el estado del modelo en un archivo checkpoint"""
        checkpoint = {
            'pred_model_type': self.PRED_MODEL_TYPE,
            'num_classes': self.num_classes,
            'torch_state_dict': self.state_dict(),
            'alpha': self.alpha,
            'q_hat': self.q_hat
        }
        torch.save(checkpoint, save_model_path)
    
    
    def load_checkpoint(self, checkpoint):
        """Carga el estado del modelo desde un checkpoint"""
        # Extrae el número de salidas del modelo guardado
        weight_key = 'classifier.fc2.2.weight'
        bias_key = 'classifier.fc2.2.bias'
        
        if weight_key in checkpoint['torch_state_dict']:
            out_features = checkpoint['torch_state_dict'][weight_key].shape[0]
            
            if out_features != self.output_size:
                checkpoint['torch_state_dict'].pop(weight_key, None)
                checkpoint['torch_state_dict'].pop(bias_key, None)
        self.load_state_dict(checkpoint['torch_state_dict'], strict=False)
        if checkpoint['pred_model_type'] == self.PRED_MODEL_TYPE and checkpoint['alpha'] == self.alpha and 'q_hat' in checkpoint:
            self.q_hat = checkpoint['q_hat']
    
    
    def calibrate(self, calib_loader):
        
        # Obtiene las clases predichas y verdaderas para el conjunto de calibración
        calib_pred_scores, calib_true_classes  = self._inference(calib_loader, return_probs=True)
        
        # Calcula el nivel de cuantificación ajustado basado en el tamaño del conjunto de calibración y alpha
        n = len(calib_true_classes)
        q_level = math.ceil((1.0 - self.alpha) * (n + 1.0)) / n
        
        # Calcula las puntuaciones de no conformidad 
        nonconformity_scores = 1 - calib_pred_scores[torch.arange(n), calib_true_classes]
        
        # Calcula el cuantil empírico q_hat que se usará para formar los conjuntos de predicción
        self.q_hat = torch.quantile(nonconformity_scores, q_level, interpolation='higher')
    
    
    def inference(self, dataloader):
        
        # Lanza un error si el modelo no está calibrado (q_hat no está determinado)
        if self.q_hat is None:
            raise ValueError("Modelo no calibrado")
        
        # Obtiene las predicciones y las clases verdaderas para el conjunto de evaluación
        pred_scores, true_classes = self._inference(dataloader, return_probs=True)
        
        # Determina la clase  predicha como la clase  con mayor puntuación
        _, pred_classes = torch.max(pred_scores, dim=1) 
        
        # Construye el conjunto de predicción seleccionando las clases con score >= 1 - qhat
        pred_sets = (pred_scores >= (1 - self.q_hat)).to(torch.uint8)  # (n, num_classes)
        
        # Asegura que ninguna muestra tenga conjunto vacío
        empty_rows = (pred_sets.sum(dim=1) == 0)  # (n,)
        pred_sets[empty_rows, :] = 1
        
        # Devuelve las clases predichas, conjuntos de predicción y clases verdaderas
        return pred_classes, pred_sets, true_classes

#-------------------------------------------------------------------------------------------------------------

class ResNeXtClassifier_MCM(ResNeXtClassifier):
    
    PRED_MODEL_TYPE = 'MCM'
    
    def __init__(self, num_classes, confidence=0.9):
        
        # Inicializa la clase padre con los parámetros base
        super().__init__(num_classes)
        
        # Parámetros para la conformal prediction
        self.alpha = 1-confidence
        self.q_hat_per_class = {}
    
    
    def save_checkpoint(self, save_model_path):
        """Guarda el estado del modelo en un archivo checkpoint"""
        checkpoint = {
            'pred_model_type': self.PRED_MODEL_TYPE,
            'num_classes': self.num_classes,
            'torch_state_dict': self.state_dict(),
            'alpha': self.alpha,
            'q_hat_per_class': self.q_hat_per_class
        }
        torch.save(checkpoint, save_model_path)
    
    
    def load_checkpoint(self, checkpoint):
        """Carga el estado del modelo desde un checkpoint"""
        # Extrae el número de salidas del modelo guardado
        weight_key = 'classifier.fc2.2.weight'
        bias_key = 'classifier.fc2.2.bias'
        
        if weight_key in checkpoint['torch_state_dict']:
            out_features = checkpoint['torch_state_dict'][weight_key].shape[0]
            
            if out_features != self.output_size:
                checkpoint['torch_state_dict'].pop(weight_key, None)
                checkpoint['torch_state_dict'].pop(bias_key, None)
        
        self.load_state_dict(checkpoint['torch_state_dict'], strict=False)
        if (checkpoint['pred_model_type'] == self.PRED_MODEL_TYPE and 
            checkpoint['alpha'] == self.alpha and 
            'q_hat_per_class' in checkpoint
        ):
            self.q_hat_per_class = checkpoint['q_hat_per_class']
    
    
    def calibrate(self, calib_loader):
        
        #
        calib_pred_scores, calib_true_classes = self._inference(calib_loader, return_probs=True)
        n = len(calib_true_classes)
        
        # Inicializa estructura para guardar puntuaciones por clase verdadera
        scores_by_class = {k: [] for k in range(self.num_classes)}
        
        for i in range(n):
            true_label = calib_true_classes[i].item()
            score = 1 - calib_pred_scores[i, true_label].item()
            scores_by_class[true_label].append(score)

        for cls, scores in scores_by_class.items():
            if len(scores) == 0:
                self.q_hat_per_class[cls] = 1.0  # valor alto para evitar predicción sobre esta clase
                continue
            # Calcula el cuantíl empírico para la clase
            q_level = math.ceil((1.0 - self.alpha) * (len(scores) + 1)) / len(scores)
            self.q_hat_per_class[cls] = torch.quantile(
                torch.tensor(scores), q_level, interpolation="higher"
            )
    
    
    def inference(self, dataloader):
        
        if not self.q_hat_per_class:
            raise ValueError("Modelo no calibrado")
        
        pred_scores, true_classes = self._inference(dataloader, return_probs=True)
        _, pred_classes = torch.max(pred_scores, dim=1)
        
        # Construye conjunto de predicción usando q_hat específico por clase
        n = pred_scores.shape[0]
        pred_sets = torch.zeros_like(pred_scores, dtype=torch.uint8)
        
        for c in range(self.num_classes):
            threshold = 1 - self.q_hat_per_class.get(c, 1.0)  # por defecto no incluye
            pred_sets[:, c] = (pred_scores[:, c] >= threshold).to(torch.uint8)
        
        # Evita conjuntos vacíos
        empty_rows = (pred_sets.sum(dim=1) == 0)
        pred_sets[empty_rows,:] = 1
        
        return pred_classes, pred_sets, true_classes

#-------------------------------------------------------------------------------------------------------------

class ResNeXtClassifier_APS(ResNeXtClassifier):
    
    PRED_MODEL_TYPE = 'APS'
    
    def __init__(self, num_classes, confidence=0.9, random=False):
        
        # Inicializa la clase padre con los parámetros base
        super().__init__(num_classes)
        
        # Parámetros para la conformal prediction
        self.alpha = 1-confidence
        self.q_hat = None 
        
        self.random = random
    
    
    def save_checkpoint(self, save_model_path):
        """Guarda el estado del modelo en un archivo checkpoint"""
        checkpoint = {
            'pred_model_type': self.PRED_MODEL_TYPE,
            'num_classes': self.num_classes,
            'torch_state_dict': self.state_dict(),
            'alpha': self.alpha,
            'q_hat': self.q_hat,
            'random': self.random,
        }
        torch.save(checkpoint, save_model_path)
    
    
    def load_checkpoint(self, checkpoint):
        """Carga el estado del modelo desde un checkpoint"""
        # Extrae el número de salidas del modelo guardado
        weight_key = 'classifier.fc2.2.weight'
        bias_key = 'classifier.fc2.2.bias'
        
        if weight_key in checkpoint['torch_state_dict']:
            out_features = checkpoint['torch_state_dict'][weight_key].shape[0]
            
            if out_features != self.output_size:
                checkpoint['torch_state_dict'].pop(weight_key, None)
                checkpoint['torch_state_dict'].pop(bias_key, None)
        
        self.load_state_dict(checkpoint['torch_state_dict'], strict=False)
        
        if (checkpoint['pred_model_type'] == self.PRED_MODEL_TYPE and 
            checkpoint['alpha'] == self.alpha and 
            'q_hat' in checkpoint
        ):
            self.q_hat = checkpoint['q_hat']
    
    
    def calibrate(self, calib_loader, random=None):
        
        #
        random = self.random if random is None else random
        
        # Obtiene las clases predichas y verdaderas para el conjunto de calibración
        pred_scores, true_labels  = self._inference(calib_loader, return_probs=True)
        
        # Obtiene el número de instancias del conjunto
        n = len(true_labels)
        
        # Ordena el ranking de scores con los índices permutados de clases y calcula la suma acumulada
        sorted_scores, sorted_class_perm_index = torch.sort(pred_scores, dim=1, descending=True)
        cum_sorted_scores = torch.cumsum(sorted_scores)
        
        # Obtiene el índice de la etiqueta verdadera en en el ranking de cada instancia
        matches = (sorted_class_perm_index == true_labels.unsqueeze(1))
        true_class_rank = matches.float().argmax(dim=1)
        
        # Recolecta los scores y suma acumulada en la posición del índice verdadero
        true_score = sorted_scores[torch.arange(n), true_class_rank]
        true_cumscore = cum_sorted_scores[torch.arange(n), true_class_rank]
        
        #
        if random:
            # Genera un vector de valores aleatorios entre 0 y 1, uno por instancia 
            U = torch.rand(n)
            
            # Calcula V
            V = (true_cumscore - (1-self.alpha)) / true_score
            
            # Calcula nonconformity_scores con indexado condicional
            # Si U > V → usar cumulative_scores en true_class_rank
            # Si U <= V → usar cumulative_scores en true_class_rank - 1
            
            # Evita valores negativos en el índice
            adjusted_rank = torch.clamp(true_class_rank - 1, min=0)
            
            # Selección de valores según condición
            nonconformity_scores = torch.where(
                U <= V,
                cum_sorted_scores[torch.arange(n), adjusted_rank],
                true_cumscore
            )
        
        else:
            
            #
            nonconformity_scores = true_cumscore
        
        # Calcula el nivel de cuantificación ajustado basado en el tamaño del conjunto de calibración y alpha
        q_level = math.ceil((1.0 - self.alpha) * (n + 1.0)) / n
        
        # Calcula el umbral de no conformidad como el percentil q_level de las puntuaciones de no conformidad
        self.q_hat = torch.quantile(nonconformity_scores, q_level, interpolation='higher').item()
    
    
    def inference(self, dataloader, random=None):
        
        # Lanza un error si el modelo no está calibrado (q_hat no está determinado)
        if self.q_hat is None:
            raise ValueError("Modelo no calibrado")
        
        #
        random = self.random if random is None else random
        
        # Obtiene las predicciones y las clases verdaderas para el conjunto de evaluación
        pred_scores, true_labels = self._inference(dataloader, return_probs=True)
        
        # Obtiene el número de instancias del conjunto y el número de clases
        n, num_classes = pred_scores.shape
        
        # Determina la clase predicha como la clase con mayor puntuación
        _, pred_labels = torch.max(pred_scores, dim=1) 
        
        # Ordena los scores de mayor a menor y calcula la suma acumulada
        sorted_scores, sorted_class_perm_index = torch.sort(pred_scores, dim=1, descending=True)
        cum_sorted_scores = torch.cumsum(sorted_scores, dim=1)
        
        # Obtiene el índice (0-indexed) de la última clase en el ranking que no supera el umbral de no conformidad
        matches = cum_sorted_scores <= self.q_hat
        has_match = matches.any(dim=1)
        last_class_ranks = torch.where(
            has_match, 
            matches.int().sum(dim=1)-1,
            torch.ones(n, dtype=torch.uint8)
        )
        
        if random:
            # Genera un vector de valores aleatorios entre 0 y 1, uno por instancia 
            U = torch.rand(n)
            
            #
            last_score = sorted_scores[torch.arange(n), last_class_ranks]
            last_cum_scores = cum_sorted_scores[torch.arange(n), last_class_ranks]
            
            #
            V = (last_cum_scores - self.q_hat) / last_score
            
            #
            last_class_ranks = torch.where(
                (U <= V) & (last_class_ranks>=1),
                last_class_ranks-1,
                last_class_ranks
            )
        
        #
        idx = torch.arange(num_classes)
        inclusion_mask = idx < last_class_ranks.unsqueeze(1)
        
        #
        pred_sets = torch.zeros_like(pred_scores, dtype=torch.uint8)
        pred_sets.scatter_(1, sorted_class_perm_index, inclusion_mask.to(torch.uint8))
        
        return pred_labels, pred_sets, true_labels


#-------------------------------------------------------------------------------------------------------------

class ResNeXtClassifier_RAPS(ResNeXtClassifier):
    
    PRED_MODEL_TYPE = 'RAPS'
    
    def __init__(self, num_classes, confidence=0.9, lambda_reg=0.0, k_reg=0, random=False):
        
        # Inicializa la clase padre con los parámetros base
        super().__init__(num_classes)
        
        # Parámetros para la conformal prediction
        self.alpha = 1 - confidence
        self.q_hat = None
        self.lambda_reg = lambda_reg
        self.k_reg = k_reg
        
        self.random = random
    
    
    def save_checkpoint(self, save_model_path):
        """Guarda el estado del modelo en un archivo checkpoint"""
        checkpoint = {
            'pred_model_type': self.PRED_MODEL_TYPE,
            'num_classes': self.num_classes,
            'torch_state_dict': self.state_dict(),
            'alpha': self.alpha,
            'q_hat': self.q_hat,
            'lambda_reg': self.lambda_reg,
            'k_reg': self.k_reg,
            'random': self.random
        }
        torch.save(checkpoint, save_model_path)
    
    
    def load_checkpoint(self, checkpoint):
        """Carga el estado del modelo desde un checkpoint"""
        # Extrae el número de salidas del modelo guardado
        weight_key = 'classifier.fc2.2.weight'
        bias_key = 'classifier.fc2.2.bias'
        
        if weight_key in checkpoint['torch_state_dict']:
            out_features = checkpoint['torch_state_dict'][weight_key].shape[0]
            
            if out_features != self.output_size:
                print("Out features: ", out_features)
                print("Output size: ", self.output_size)
                checkpoint['torch_state_dict'].pop(weight_key, None)
                checkpoint['torch_state_dict'].pop(bias_key, None)
        
        self.load_state_dict(checkpoint['torch_state_dict'], strict=False)
        
        if (checkpoint['pred_model_type'] == self.PRED_MODEL_TYPE and 
            checkpoint['alpha'] == self.alpha and 
            'q_hat' in checkpoint
        ):
            self.q_hat = checkpoint['q_hat']
            self.lambda_reg = checkpoint.get('lambda_reg', 0.0)
            self.k_reg = checkpoint.get('k_reg', 0)
    
    
    @staticmethod
    def _calibrate_RAPS(
        pred_scores: torch.Tensor,
        true_labels: torch.Tensor,
        lmbda: float,
        k_reg: int,
        alpha: float,
        random: bool = False
    ) -> float:
        
        # Usa el valor por defecto de `self.random` si no se especifica explícitamente
        random = random if random is not None else self.random
        
        # Obtiene el número de instancias del conjunto y el número de clases
        n, num_classes = pred_scores.shape
        
        # Ordena los scores de cada instancia de mayor a menor y guarda los índices de ordenamiento
        sorted_scores, sorted_class_perm_index = torch.sort(pred_scores, dim=1, descending=True)
        # Calcula la suma acumulada de los scores ordenados
        cum_sorted_scores = torch.cumsum(sorted_scores, dim=1)
        
        # Crea un vector de penalización: sin penalización hasta k_reg, luego aplica penalización constante λ
        penalties = torch.zeros((1, num_classes))
        penalties[0, k_reg:] += lmbda
        # Calcula la penalización acumulada
        cumulative_penalties = torch.cumsum(penalties, dim=1)
        
        # Encuentra, para cada instancia, la posición (0-indexed) de la clase verdadera en el ranking ordenado 
        matches = (sorted_class_perm_index==true_labels.unsqueeze(1))
        true_class_rank = matches.int().argmax(dim=1)
        
        # Obtiene el score de la clase verdadera para cada instancia y la suma acumulada
        true_score = sorted_scores[torch.arange(n), true_class_rank]
        true_cum_score = cum_sorted_scores[torch.arange(n), true_class_rank]
        
        # Obtiene la penalización para la clase verdadera y la suma acumulada
        true_penalty = penalties[0, true_class_rank]
        true_cum_penalty = cumulative_penalties[0, true_class_rank]
        
        if random:
        
            # Genera un vector de valores aleatorios entre 0 y 1, uno por instancia 
            U = torch.rand(n)
            
            # Calcula los nonconformity scores ajustando con ruido aleatorio (si la clase verdadera no está al principio)
            nonconformity_scores = torch.where(
                true_class_rank >= 1,
                true_cum_score + true_cum_penalty - U * true_score,
                true_cum_score + true_cum_penalty
            )
        
        else:
            
            # Nonconformity scores sin aleatoriedad
            nonconformity_scores = true_cum_score + true_cum_penalty
        
        
        # Calcula el nivel de cuantificación ajustado basado en el tamaño del conjunto de calibración y alpha
        q_level = math.ceil((1.0 - alpha) * (n + 1.0)) / n
        
        # Calcula el umbral de no conformidad como el percentil q_level de las puntuaciones de no conformidad
        q_hat = torch.quantile(nonconformity_scores, q_level, interpolation='higher').item()
        
        return q_hat
    
    
    @staticmethod
    def _inference_RAPS(
        pred_scores : torch.Tensor, 
        q_hat : float, 
        lmbda : float, 
        k_reg : int,
        random : bool = False
    ) -> torch.Tensor:
        
        # Obtiene el número de instancias del conjunto y el número de clases
        n, num_classes = pred_scores.shape
        
        # Ordena los scores de mayor a menor y calcula la suma acumulada
        sorted_scores, sorted_class_perm_index = torch.sort(pred_scores, dim=1, descending=True)
        cum_sorted_scores = torch.cumsum(sorted_scores, dim=1)
        
        # Crea un vector de penalización acumulada para cada posición 
        penalties = torch.zeros((1, num_classes))
        penalties[0, k_reg:] += lmbda
        cumulative_penalties = torch.cumsum(penalties, dim=1)
        
        #
        matches = cum_sorted_scores <= q_hat
        has_match = matches.any(dim=1)
        last_class_ranks = torch.where(
            has_match,
            matches.int().sum(dim=1)-1,
            torch.ones(n, dtype=torch.uint8)
        )
        
        #
        if random:
            # Genera un vector de valores aleatorios entre 0 y 1, uno por instancia 
            U = torch.rand(n)
            
            #
            last_score = sorted_scores[torch.arange(n), last_class_ranks]
            last_cum_scores = cum_sorted_scores[torch.arange(n), last_class_ranks]
            
            #
            last_penalty = penalties[0, last_class_ranks]
            last_cum_penalty = cumulative_penalties[0, last_class_ranks]
            
            #
            V = (last_cum_scores + last_cum_penalty - q_hat) / (last_score + last_penalty)
            
            #
            last_class_ranks = torch.where(
                (U <= V) & (last_class_ranks>=1),
                last_class_ranks-1,
                last_class_ranks
            )

        #
        idx = torch.arange(num_classes)
        inclusion_mask = idx <= last_class_ranks.unsqueeze(1)
        
        # 
        pred_sets = torch.zeros_like(pred_scores, dtype=torch.uint8)
        pred_sets.scatter_(1, sorted_class_perm_index, inclusion_mask.to(torch.uint8))
        
        return pred_sets
    
    
    def calibrate(self, calib_loader, random=None):
        
        # Usa el valor por defecto de `self.random` si no se especifica explícitamente
        random = random if random is not None else self.random
        
        # Obtiene las clases predichas y verdaderas para el conjunto de calibración
        pred_scores, true_labels  = self._inference(calib_loader, return_probs=True)
        
        #
        self.q_hat = self._calibrate_RAPS(pred_scores, true_labels, self.lambda_reg, self.k_reg, self.alpha, random)
    
    
    def inference(self, dataloader, random=None):
        
        # Usa el valor por defecto de `self.random` si no se especifica explícitamente
        random = random if random is not None else self.random
        
        # Lanza un error si el modelo no está calibrado (q_hat no está determinado)
        if self.q_hat is None:
            raise ValueError("Modelo no calibrado")
        
        # Obtiene las predicciones y las clases verdaderas para el conjunto de evaluación
        pred_scores, true_labels = self._inference(dataloader, return_probs=True)
        
        # Determina la clase predicha como la clase con mayor score
        _, pred_labels = torch.max(pred_scores, dim=1) 
        
        #
        pred_sets = self._inference_RAPS(pred_scores, self.q_hat, self.lambda_reg, self.k_reg, random)
        
        return pred_labels, pred_sets, true_labels
    
    
    def _get_kstar(self, pred_scores, true_labels):
        """
        Calcula el valor óptimo de k_reg (k*), que ...
        """
        # Obtiene el número de instancias del conjunto
        n = len(true_labels)

        # Obtiene el ranking descendente de las predicciones
        sorted_pred_index = torch.argsort(pred_scores, dim=1, descending=True)

        # Encuentra la posición (0-indexed) de la clase verdadera en el ranking
        true_label_idx = (sorted_pred_index == true_labels.unsqueeze(1)).nonzero(as_tuple=False)[:,1]
        
        # Calcula el cuantil ajustado
        k_level = math.ceil((1.0 - self.alpha) * (n + 1.0)) / n
        khat = torch.quantile(true_label_idx.float(), k_level, interpolation='higher').item() + 1

        return int(khat)
    
    
    def _get_lambda(self, pred_scores, true_labels, k_reg, random=None):
        """
        Busca el valor de lambda que minimiza el tamaño medio del conjunto predictivo,
        manteniendo cobertura conforme a RAPS.
        """
        
        # Usa el valor por defecto de `self.random` si no se especifica explícitamente
        random = random if random is not None else self.random
        
        #
        def objetive_log(log_lmbda):
            lmbda = 10 ** log_lmbda
            q_hat = self._calibrate_RAPS(pred_scores, true_labels, lmbda, k_reg, self.alpha, random)
            pred_sets = self._inference_RAPS(pred_scores, q_hat, lmbda, k_reg, random)
            return mean_set_size(pred_sets)
        
        #
        result = minimize_scalar(objetive_log, bounds=(-3,2), method='bounded', options={'maxiter': 1000})
        
        # Devuelve el valor óptimo en escala lineal
        return (10** result.x)
    
    
    def auto_configure(self, dataloader):
        
        # Pasa por el modelo y obtiene scores y etiquetas verdaderas
        pred_scores, true_labels = self._inference(dataloader, return_probs=True)
        
        #
        self.k_reg = self._get_kstar(pred_scores, true_labels)
        print("k_reg: ", self.k_reg)
        self.lambda_reg = self._get_lambda(pred_scores, true_labels, self.k_reg)
        print("lambda: ", self.lambda_reg)


#-------------------------------------------------------------------------------------------------------------

class ResNeXtClassifier_SAPS(ResNeXtClassifier):
    
    PRED_MODEL_TYPE = 'SAPS'
    
    def __init__(self, num_classes, confidence=0.9):
        
        # Inicializa la clase padre con los parámetros base
        super().__init__(num_classes)
        
        # Parámetros para la conformal prediction
        self.alpha = 1 - confidence
        self.lambda_reg = None
        self.q_hat = None
    
    
    def save_checkpoint(self, save_model_path):
        """Guarda el estado del modelo en un archivo checkpoint"""
        
        checkpoint = {
            'pred_model_type': self.PRED_MODEL_TYPE,
            'num_classes': self.num_classes,
            'torch_state_dict': self.state_dict(),
            'alpha': self.alpha,
            'lambda_reg': self.lambda_reg,
            'q_hat': self.q_hat,
        }
        torch.save(checkpoint, save_model_path)
    
    
    def load_checkpoint(self, checkpoint):
        """Carga el estado del modelo desde un checkpoint"""
        
        # Extrae el número de salidas del modelo guardado
        weight_key = 'classifier.fc2.2.weight'
        bias_key = 'classifier.fc2.2.bias'
        
        if weight_key in checkpoint['torch_state_dict']:
            out_features = checkpoint['torch_state_dict'][weight_key].shape[0]
            
            if out_features != self.output_size:
                print("Out features: ", out_features)
                print("Output size: ", self.output_size)
                checkpoint['torch_state_dict'].pop(weight_key, None)
                checkpoint['torch_state_dict'].pop(bias_key, None)
        
        self.load_state_dict(checkpoint['torch_state_dict'], strict=False)
        
        if (checkpoint['pred_model_type'] == self.PRED_MODEL_TYPE and 
            checkpoint['alpha'] == self.alpha and 
            'q_hat' in checkpoint
        ):
            self.q_hat = checkpoint.get('q_hat', None)
            self.lambda_reg = checkpoint.get('lambda_reg', None)
    
    
    @staticmethod
    def _calibrate_SAPS(
        pred_scores: torch.Tensor,
        true_labels: torch.Tensor,
        lmbda: float,
        alpha: float
    ) -> float:
        
        # Obtiene el número de instancias del conjunto y el número de clases
        n, num_classes = pred_scores.shape
        
        # Ordena los scores de cada instancia de mayor a menor y guarda los índices de ordenamiento
        sorted_scores, sorted_class_perm_index = torch.sort(pred_scores, dim=1, descending=True)
        
        # Encuentra, para cada instancia, la posición (0-indexed) de la clase verdadera en el ranking ordenado 
        matches = (sorted_class_perm_index==true_labels.unsqueeze(1))
        true_class_rank = matches.int().argmax(dim=1)
        
        #
        max_score = sorted_scores[torch.arange(n), 0]
        
        # Genera un vector de valores aleatorios entre 0 y 1, uno por instancia 
        U = torch.rand(n)
        
        # Calcula los nonconformity scores ajustando con ruido aleatorio (si la clase verdadera no está al principio)
        nonconformity_scores = torch.where(
            true_class_rank >= 1,
            max_score + (true_class_rank-1+U)*lmbda, 
            max_score * U
        )
        
        # Calcula el nivel de cuantificación ajustado basado en el tamaño del conjunto de calibración y alpha
        q_level = math.ceil((1.0 - alpha) * (n + 1.0)) / n
        
        # Calcula el umbral de no conformidad como el percentil q_level de las puntuaciones de no conformidad
        q_hat = torch.quantile(nonconformity_scores, q_level, interpolation='higher').item()
        
        return q_hat
    
    
    @staticmethod
    def _inference_SAPS(
        pred_scores : torch.Tensor, 
        q_hat : float, 
        lmbda : float
    ) -> torch.Tensor:
        
        # Obtiene el número de instancias del conjunto y el número de clases
        n, num_classes = pred_scores.shape
        
        # Ordena los scores de mayor a menor ...
        sorted_scores, sorted_class_perm_index = torch.sort(pred_scores, dim=1, descending=True)
        
        # Obtiene el máximo score de predicción para cada instancia
        max_score = sorted_scores[torch.arange(n), 0]
        
        # Crea una matriz con índices de clase (0 a num_classes-1) para cada instancia
        pos_matrix = torch.arange(num_classes).expand(n, num_classes)

        # Genera una matriz de valores aleatorios entre 0 y 1 para cada instancia y clase
        U_matrix = torch.rand(n).unsqueeze(1).expand(-1, num_classes)

        # Calcula los scores de no conformidad para cada clase de cada instancia
        nonconformity_scores = torch.where(
            pos_matrix == 0,
            max_score.unsqueeze(1) * U_matrix, 
            max_score.unsqueeze(1) + ((pos_matrix - 1 + U_matrix) * lmbda)
        )
        
        # Crea una máscara indicando qué clases deben incluirse en el conjunto de predicción de cada instancia
        inclusion_mask = nonconformity_scores <= q_hat
        
        # Convierte la máscara de inclusión a un formato one-hot
        pred_sets = torch.zeros_like(pred_scores, dtype=torch.uint8)
        pred_sets.scatter_(1, sorted_class_perm_index, inclusion_mask.to(torch.uint8))
        
        return pred_sets
    
    
    def calibrate(self, calib_loader, random=None):
        
        # Usa el valor por defecto de `self.random` si no se especifica explícitamente
        random = random if random is not None else self.random
        
        # Obtiene las clases predichas y verdaderas para el conjunto de calibración
        pred_scores, true_labels  = self._inference(calib_loader, return_probs=True)
        
        # Obtiene el umbral de no conformidad 
        self.q_hat = self._calibrate_SAPS(pred_scores, true_labels, self.lambda_reg, self.alpha)
    
    
    def inference(self, dataloader, random=None):
        
        # Usa el valor por defecto de `self.random` si no se especifica explícitamente
        random = random if random is not None else self.random
        
        # Lanza un error si el modelo no está calibrado (q_hat no está determinado)
        if self.q_hat is None:
            raise ValueError("Modelo no calibrado")
        
        # Obtiene las predicciones y las clases verdaderas para el conjunto de evaluación
        pred_scores, true_labels = self._inference(dataloader, return_probs=True)
        
        # Determina la clase predicha como la clase con mayor score
        _, pred_labels = torch.max(pred_scores, dim=1) 
        
        # Infiere los conjuntos de predicción conformal 
        pred_sets = self._inference_SAPS(pred_scores, self.q_hat, self.lambda_reg)
        
        return pred_labels, pred_sets, true_labels
    
    
    def _get_lambda(self, pred_scores, true_labels):
        """
        Busca el valor de lambda que minimiza el tamaño medio del conjunto predictivo.
        """
        
        def objective_log(log_lmbda):
            lmbda = 10 ** log_lmbda  # transformar a escala lineal
            q_hat = self._calibrate_SAPS(pred_scores, true_labels, lmbda, self.alpha)
            pred_sets = self._inference_SAPS(pred_scores, q_hat, lmbda)
            return mean_set_size(pred_sets)
        
        # Optimiza lambda en escala logarítmica para cubrir un rango amplio
        result = minimize_scalar(objective_log, bounds=(-3, 2), method='bounded', options={'maxiter': 1000})
        
        # Devuelve el valor óptimo en escala lineal
        return (10** result.x)
    
    
    def auto_configure(self, dataloader):
        
        # Pasa por el modelo y obtiene scores y etiquetas verdaderas
        pred_scores, true_labels = self._inference(dataloader, return_probs=True)
        
        # Encuentra el valor óptimo de lambda
        self.lambda_reg = self._get_lambda(pred_scores, true_labels, self.k_reg)
        print("lambda: ", self.lambda_reg)


#-------------------------------------------------------------------------------------------------------------
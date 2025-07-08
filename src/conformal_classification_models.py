#-------------------------------------------------------------------------------------------------------------
# BIBLIOTECAS ------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torchvision 
import math

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
        
        super().__init__()
        
        # Almacena el número de clases de clasificación
        self.num_classes = num_classes
        
        # Define las componentes de la red: 
        # 1) Extractor de características, pooling y flattening 
        self.feature_extractor = FeatureExtractorResNeXt()
        self.pool_avg = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()
        
        # 2) Metadata embedding
        self.embedding = nn.Sequential(
            nn.Embedding(num_embeddings=2, embedding_dim=16),
            nn.LayerNorm(16)  # Normalización para embeddings
        )
        
        # 3) Classifier según el tipo de clasificación
        input_size = 2048 + 16 # características aplanadas + embedding
        output_size = 1 if num_classes==2 else num_classes
        self.classifier = ClassifierResNeXt(input_size, output_size)
        
        # Define la función de pérdida
        self.loss_function = nn.BCEWithLogitsLoss() if num_classes==2 else nn.CrossEntropyLoss()
    
    
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
        self.load_state_dict(checkpoint['torch_state_dict'])
    
    
    def forward(self, image, metadata):
        """Paso forward del modelo"""
        
        # Extrae y aplana las características
        x = self.feature_extractor(image)
        x = self.pool_avg(x)
        x = self.flatten(x)
        
        # Procesa los metadatos con el embedding
        y = self.embedding(metadata)
        
        # Concatena las características profundas aplanadas con el embedding de sexo 
        z = torch.cat([x,y], dim=1)
        
        # Pasa por el clasificador para obtener las predicciones
        outputs =  self.classifier(z)
        
        # Ajusta la dimensión de salida si es necesario
        outputs = outputs.squeeze(-1) if outputs.dim() > 1 and outputs.shape[-1] == 1 else outputs

        return outputs


    def get_layer_groups(self):
        """Devuelve los parámetros del modelo agrupados por capas, de más superficiales a más profundas"""
        
        layer_groups = []
        
        layer_groups.append(list(self.feature_extractor.conv1.parameters()))
        layer_groups.append(list(self.feature_extractor.conv2.parameters()))
        layer_groups.append(list(self.feature_extractor.conv3.parameters()))
        layer_groups.append(list(self.feature_extractor.conv4.parameters()))
        layer_groups.append(list(self.feature_extractor.conv5.parameters()))
        layer_groups.append(list(self.classifier.fc1.parameters(), self.embedding.parameters()))
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
        for images, metadata, labels in dataloader:
            
            # Obtiene las imágenes y metadata de entrenamiento y sus valores objetivo
            images, metadata, labels = images.to('cuda'), metadata.to('cuda'), labels.to('cuda')
            
            # Limpia los gradientes de la iteración anterior
            optimizer.zero_grad()
            
            # Obtiene las predicciones del modelo
            outputs = self.forward(images, metadata)
            
            # Asegura que las etiquetas tengan el tipo correcto
            labels = labels.float()
            
            # Calcula la pérdida de las predicciones
            loss = loss_fn(outputs, labels)
            
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
    
    
    def _inference(self, dataloader):
        
        # Pone la red en modo evaluación 
        self.eval()
        
        # Inicializa listas si son requeridas
        all_targets = [] 
        all_outputs = []
        
        # Flag para detectar si el dataloader entrega targets
        has_targets = False
        
        # Desactiva el cálculo de gradientes para eficiencia
        with torch.no_grad():
            for batch in dataloader:
                
                # Verifica si el batch contiene (images, metadata, targets) o solo (images, metadata)
                if isinstance(batch, (list, tuple)) and len(batch) == 3:
                    images, metadata, targets = batch 
                    images, metadata = images.to('cuda'), metadata.to('cuda')
                    has_targets = True
                    all_targets.append(targets.cpu())
                    
                elif isinstance(batch, (list, tuple)) and len(batch) == 2:
                    images, metadata = batch 
                    images, metadata = images.to('cuda'), metadata.to('cuda')
                    
                else:
                    images = batch.to('cuda')
                    metadata = None

                # Forward pass
                outputs = self.forward(images, metadata)
                
                # Convertir salidas a probabilidades
                if self.num_classes == 2:
                    # Para clasificación binaria: aplica sigmoide y crea formato (n, 2)
                    p = torch.sigmoid(outputs)
                    probabilities = torch.stack([1 - p, p], dim=1).squeeze()  # (n, 2)
                else:
                    # Para clasificación multiclase: aplica softmax normal
                    probabilities = torch.softmax(outputs, dim=1)
                
                # Recolecta las probabilidades
                all_outputs.append(probabilities.cpu())
        
        # Concatena los resultados
        outputs = torch.cat(all_outputs)
        targets = torch.cat(all_targets) if has_targets else None
        
        # Devuelve las probabilidades predichas para cada clase y las clases verdaderas
        return outputs, targets 
    
    
    def inference(self, dataloader):
        
        # Obtiene las probabilidades predichas para cada clase y la clase verdadera para cada instancia
        pred_scores, true_classes = self._inference(dataloader)
        
        # Determina la clase predicha (la de mayor probabilidad) para cada instancia
        _, pred_classes = torch.max(pred_scores, dim=1)
        
        # Convierte las clases predichas a one-hot encoding
        pred_sets = nn.functional.one_hot(pred_classes, num_classes=self.num_classes)
        
        return pred_classes, pred_sets, true_classes


    def evaluate(self, dataloader, metric_fn=None):
        """Evalúa el modelo en un conjunto de datos"""
        
        # Determina la función de métrica
        metric_fn = metric_fn if metric_fn is not None else self.loss_function 
        
        # Obtiene todas las predicciones y valores verdaderos
        all_predicted, all_targets = self._inference(dataloader)
        
        if isinstance(metric_fn, nn.BCEWithLogitsLoss):
            # Asegura que targets tengan dtype correcto (float) 
            all_targets = all_targets.float()
        
        # Calcula el valor de la métrica y lo devuelve
        return metric_fn(all_predicted, all_targets)


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
        self.load_state_dict(checkpoint['torch_state_dict'])
        if checkpoint['pred_model_type'] == self.PRED_MODEL_TYPE and checkpoint['alpha'] == self.alpha and 'q_hat' in checkpoint:
            self.q_hat = checkpoint['q_hat']
    
    
    def calibrate(self, calib_loader):
        
        # Obtiene las clases predichas y verdaderas para el conjunto de calibración
        calib_pred_scores, calib_true_classes  = self._inference(calib_loader)
        
        # Calcula el nivel de cuantificación ajustado basado en el tamaño del conjunto de calibración y alpha
        n = len(calib_true_classes)
        q_level = math.ceil((1.0 - self.alpha) * (n + 1.0)) / n
        print("q_level: ", q_level)
        
        # Calcula las puntuaciones de no conformidad 
        nonconformity_scores = 1 - calib_pred_scores[torch.arange(n), calib_true_classes]
        print("noncoformity_scores: ", nonconformity_scores)
        
        # Calcula el cuantil empírico q_hat que se usará para formar los conjuntos de predicción
        self.q_hat = torch.quantile(nonconformity_scores, q_level, interpolation='higher')
        print("q_hat: ", self.q_hat)
    
    
    def inference(self, dataloader):
        
        # Lanza un error si el modelo no está calibrado (q_hat no está determinado)
        if self.q_hat is None:
            raise ValueError("Modelo no calibrado")
        
        # Obtiene las predicciones y las clases verdaderas para el conjunto de evaluación
        pred_scores, true_classes = self._inference(dataloader)
        
        # Determina la clase  predicha como la clase  con mayor puntuación
        _, pred_classes = torch.max(pred_scores, dim=1) 
        
        # Construye el conjunto de predicción seleccionando las clases con score >= 1 - qhat
        pred_sets = (pred_scores >= (1 - self.q_hat)).to(torch.uint8)  # (n, num_classes)
        
        # Asegura que ninguna muestra tenga conjunto vacío
        empty_rows = (pred_sets.sum(dim=1) == 0)  # (n,)
        pred_sets[empty_rows, pred_classes[empty_rows]] = 1
        
        # Devuelve las clases predichas, conjuntos de predicción y clases verdaderas
        return pred_classes, pred_sets, true_classes
    
#-------------------------------------------------------------------------------------------------------------
    
class ResNeXtClassifier_Mondrian(ResNeXtClassifier):
    
    PRED_MODEL_TYPE = 'mondrian'
    
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
        self.load_state_dict(checkpoint['torch_state_dict'])
        if checkpoint['pred_model_type'] == self.PRED_MODEL_TYPE and checkpoint['alpha'] == self.alpha and 'q_hat_per_class' in checkpoint:
            self.q_hat_per_class = checkpoint['q_hat_per_class']
    
    
    def calibrate(self, calib_loader):
        
        #
        calib_pred_scores, calib_true_classes = self._inference(calib_loader)
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
            print(f"Clase {cls} -> q_hat: {self.q_hat_per_class[cls]}")
    

    def inference(self, dataloader):
        
        if not self.q_hat_per_class:
            raise ValueError("Modelo no calibrado")
        
        pred_scores, true_classes = self._inference(dataloader)
        _, pred_classes = torch.max(pred_scores, dim=1)

        # Construye conjunto de predicción usando q_hat específico por clase
        n = pred_scores.shape[0]
        pred_sets = torch.zeros_like(pred_scores, dtype=torch.uint8)
        
        for c in range(self.num_classes):
            threshold = 1 - self.q_hat_per_class.get(c, 1.0)  # por defecto no incluye
            pred_sets[:, c] = (pred_scores[:, c] >= threshold).to(torch.uint8)

        # Evita conjuntos vacíos
        empty_rows = (pred_sets.sum(dim=1) == 0)
        pred_sets[empty_rows, pred_classes[empty_rows]] = 1

        return pred_classes, pred_sets, true_classes
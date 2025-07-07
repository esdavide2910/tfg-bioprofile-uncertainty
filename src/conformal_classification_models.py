import torch
import torchvision 
import torch.nn as nn
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import math


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
    
    def __init__(self, input_size=4096, output_size=1):

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

        # nn.Linear siempre devuelve una salida 2D
        # Aplanamos la salida si solo hay una variable target
        return x.squeeze(-1) if x.dim() > 1 and x.shape[-1] == 1 else x


#-------------------------------------------------------------------------------------------------------------

class ResNeXtClassifier(nn.Module):
    
    def __init__(self, num_classes, use_metadata=False, meta_input_size=0, *args, **kwargs):
        
        super().__init__()
        
        if use_metadata and meta_input_size <= 0:
            raise ValueError("Si use_metadata=True, entonces meta_input_size debe ser > 0")
        if not use_metadata and meta_input_size != 0:
            raise ValueError("Si use_metadata=False, entonces meta_input_size debe ser = 0")
        
        self.use_metadata = use_metadata
        self.num_classes = num_classes
        
        # Extractor de características
        self.feature_extractor = FeatureExtractorResNeXt()
        
        #
        self.pool_avg = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()
        
        #
        if self.use_metadata:
            self.embedding = nn.Sequential(
                nn.Embedding(num_embeddings=2, embedding_dim=16),
                nn.LayerNorm(16)  # Normalización para embeddings
            )
        
        # 
        input_size = 2048 
        if use_metadata:
            input_size += 16

        #
        output_size = 1 if num_classes==2 else num_classes

        self.classifier = ClassifierResNeXt(input_size, output_size)
        
        # Define la función de pérdida
        self.loss_function = nn.BCEWithLogitsLoss() if num_classes==2 else nn.CrossEntropyLoss()
    
    
    def save_checkpoint(self, save_model_path):
        
        checkpoint = {
            'pred_model_type': 'base',
            'use_metadata': self.use_metadata,
            'num_classes': self.num_classes,
            'torch_state_dict': self.state_dict()
        }
        torch.save(checkpoint, save_model_path)


    def load_checkpoint(self, checkpoint):
        
        self.load_state_dict(checkpoint['torch_state_dict'])
        
    
    def forward(self, image, metadata=None, return_deep_features=False):
        
        # Extracción de características
        x = self.feature_extractor(image)
        x = self.pool_avg(x)
        x = self.flatten(x)
        
        # Obtener deep features con o sin metadatos
        if self.use_metadata:
            if metadata is None:
                raise ValueError("Metadatos requeridos pero no provistos.")
            
            if metadata.dim() > 1:
                metadata = metadata.squeeze(1)
            y = self.embedding(metadata)
            z = torch.cat([x,y], dim=1)
        else:
            z = x
        
        outputs =  self.classifier(z)
        outputs = outputs.squeeze(-1) if outputs.dim() > 1 and outputs.shape[-1] == 1 else outputs

        return outputs


    def get_layer_groups(self):
        
        layer_groups = []
        
        layer_groups.append(list(self.feature_extractor.conv1.parameters()))
        layer_groups.append(list(self.feature_extractor.conv2.parameters()))
        layer_groups.append(list(self.feature_extractor.conv3.parameters()))
        layer_groups.append(list(self.feature_extractor.conv4.parameters()))
        layer_groups.append(list(self.feature_extractor.conv5.parameters()))
        
        #
        layer_groups.append(list(self.classifier.fc1.parameters()))
        if self.use_metadata:
            layer_groups[-1].extend(self.embedding.parameters())
        layer_groups.append(list(self.classifier.fc2.parameters()))

        return layer_groups


    def train_epoch(self, dataloader, optimizer, scheduler=None, loss_fn=None):
        
        # Determinamos la función de pérdida
        loss_fn = loss_fn if loss_fn is not None else self.loss_function 
        
        # Pone la red en modo entrenamiento 
        self.train()
        
        # Inicializa la pérdida acumulada para esta época
        epoch_loss = 0
        
        #
        for images, metadata, labels in dataloader:
            
            # Obtiene las imágenes y metadata de entrenamiento y sus valores objetivo
            images, metadata, labels = images.to('cuda'), metadata.to('cuda'), labels.to('cuda')
            
            # Limpia los gradientes de la iteración anterior
            optimizer.zero_grad()
            
            #
            outputs = self.forward(images, metadata)
            
            #
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
        
        # Calcula la pérdida promedio de la época y la devolvemos
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

                # Ejecuta el modelo y recolecta los resultados
                outputs = self.forward(images, metadata)
                all_outputs.append(outputs.cpu())
                    
        
        #
        outputs = torch.cat(all_outputs)
        targets = torch.cat(all_targets) if has_targets else None

        # Devuelve una tupla (valores predichos, valores verdaderos)
        return outputs, targets 
    
    
    def inference(self, dataloader, return_probs=False):
        
        logits, true_values = self._inference(dataloader)
        
        if self.num_classes == 2:
            probs = torch.sigmoid(logits)
            pred_classes = (probs >= 0.5).long()
            
            if return_probs is True:
                return pred_classes, true_values, probs
            
        else:
            pred_classes = torch.argmax(logits, dim=1)
            
            if return_probs is True:
                probs = torch.softmax(logits, dim=1)
                return pred_classes, true_values, probs
        
        return pred_classes, true_values 


    def evaluate(self, dataloader, metric_fn=None):
        """
        Evalúa el modelo en un conjunto de datos.
        """
        # Determina la función de métrica
        metric_fn = metric_fn if metric_fn is not None else self.loss_function 
        
        # Obtiene todas las predicciones y valores verdaderos
        all_predicted, all_targets = self._inference(dataloader)
        
        if isinstance(metric_fn, nn.BCEWithLogitsLoss):
            # Asegurar que targets tengan dtype correcto y dimensión adecuada
            all_targets = all_targets.float()

        # Calcula el valor de la métrica y lo devuelve
        return metric_fn(all_predicted, all_targets)


#-------------------------------------------------------------------------------------------------------------
    
# class ResNeXtClassifier_LAC(ResNeXtClassifier):
    
#     def __init__(self, num_classes, use_metadata=False, meta_input_size=0):
        
#         super().__init__(num_classes, use_metadata, meta_input_size)
        
        
import torch
import torchvision 
import torch.nn as nn
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import math


# Clase de pérdida para la regresión cuantílica
# Esta pérdida mide cuán bien predice un modelo los cuantiles de una distribución
class PinballLoss(nn.Module):
    
    def __init__(self, quantiles):
        
        super().__init__()
        self.quantiles = quantiles
        
        
    def forward(self, preds, targets):
        # Asegura que los targets no estén marcados para el cálculo de gradientes (esto es importante porque 
        # solo las predicciones deben participar en el backpropagation, no las etiquetas reales)
        assert not targets.requires_grad
        # Asegura que el batch size de las predicciones y los targets coincide
        assert preds.size(0) == targets.size(0)
        # Asegura que el número de columnas en preds coincida con el número de cuantiles que se quieren 
        # predecir
        assert preds.size(1) == len(self.quantiles)
        
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = targets - preds[:,i]
            losses.append(torch.max((q-1)*errors, q*errors).unsqueeze(1))
            
        #
        all_losses = torch.cat(losses, dim=1)
        loss = torch.mean(torch.sum(all_losses, dim=1))
        return loss

#-------------------------------------------------------------------------------------------------------------

class FeatureExtractorResNeXt(nn.Module):
    
    def __init__(self):
        
        super(FeatureExtractorResNeXt, self).__init__()
        
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

# class MetadataEmbedding(nn.Module):
    
#     def __init__(self, input_size=1, layers_sizes=[32, 64]):

#         super(MetadataEmbedding, self).__init__()  
        
#         self.input_size = input_size
#         self.layers_sizes = layers_sizes
        
#         layers = []
#         in_features = input_size
        
#         for _, hidden_size in enumerate(layers_sizes):
#             layers.append(nn.Linear(in_features, hidden_size))
#             layers.append(nn.BatchNorm1d(hidden_size))
#             layers.append(nn.ReLU(inplace=True))
#             layers.append(nn.Dropout(p=0.5))
#             in_features = hidden_size
            
#         self.embedding = nn.Sequential(*layers)
    
    
#     def forward(self, x):
        
#         return self.embedding(x)
    

#     def get_last_layer_size(self):
        
#         return self.layers_sizes[-1]
    

#-------------------------------------------------------------------------------------------------------------

class ResNeXtRegressor(nn.Module):
    
    def __init__(self, use_metadata=False, meta_input_size=0, *args):
        
        super().__init__()
        
        if use_metadata and meta_input_size <= 0:
            raise ValueError("Si use_metadata=True, entonces meta_input_size debe ser > 0")
        if not use_metadata and meta_input_size != 0:
            raise ValueError("Si use_metadata=False, entonces meta_input_size debe ser = 0")
        
        self.use_metadata = use_metadata
        
        # Extractor de características
        self.feature_extractor = FeatureExtractorResNeXt()
        
        #
        self.pool_avg = nn.AdaptiveAvgPool2d((1,1))
        # self.pool_max = nn.AdaptiveMaxPool2d((1,1))
        self.flatten = nn.Flatten()
        
        #
        if self.use_metadata:
            self.embedding = nn.Sequential(
                nn.Embedding(num_embeddings=2, embedding_dim=16),
                nn.LayerNorm(16)  # Normalización para embeddings
            )
        
        # Clasificador con inicialización mejorada
        input_size = 2048 #4096  # Asumiendo que es el tamaño de cat(avg,max)
        if use_metadata:
            input_size += 16

        self.classifier = ClassifierResNeXt(input_size)
        
        # Define la función de pérdida
        self.loss_function = nn.MSELoss()
    
    
    def save_checkpoint(self, save_model_path):
        
        checkpoint = {
            'pred_model_type': 'base',
            'use_metadata': self.use_metadata,
            'torch_state_dict': self.state_dict()
        }
        torch.save(checkpoint, save_model_path)


    def load_checkpoint(self, checkpoint):
        
        self.load_state_dict(checkpoint['torch_state_dict'])
        
    
    def forward(self, image, metadata=None, return_deep_features=False):
        
        # Extracción de características
        x = self.feature_extractor(image)
        x = self.pool_avg(x)
        # avg = self.pool_avg(x)
        # max = self.pool_max(x)
        # x = torch.cat([avg, max], dim=1)
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

        deep_features = self.classifier.fc1(z)
        outputs = self.classifier.fc2(deep_features)
        outputs = outputs.squeeze(-1) if outputs.dim() > 1 and outputs.shape[-1] == 1 else outputs

        return (outputs, deep_features) if return_deep_features else outputs


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
        for images, metadata, targets in dataloader:
            
            # Obtiene las imágenes y metadata de entrenamiento y sus valores objetivo
            images, metadata, targets = images.to('cuda'), metadata.to('cuda'), targets.to('cuda')
            
            # Limpia los gradientes de la iteración anterior
            optimizer.zero_grad()
            
            #
            outputs = self.forward(images, metadata)
            
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
    
    
    def _inference(self, dataloader, include_deep_features=False):
        
        # Pone la red en modo evaluación 
        self.eval()
        
        # Inicializa listas si son requeridas
        all_targets = [] 
        all_outputs = []
        if include_deep_features:
            all_deep_features = []
        
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
                    images = batch
                    images = images.to('cuda')
                    metadata = None

                # Ejecuta el modelo y recolecta resultados
                if include_deep_features:
                    outputs, deep_features = self.forward(images, metadata, return_deep_features=True)
                    all_deep_features.append(deep_features.cpu())
                else:
                    outputs = self.forward(images, metadata)
                    
                all_outputs.append(outputs.cpu())
                
        #
        outputs = torch.cat(all_outputs)
        targets = torch.cat(all_targets) if has_targets else None
        if include_deep_features:
            deep_features = torch.cat(all_deep_features)
            return outputs, targets, deep_features

        # Devuelve una tupla (valores predichos, valores verdaderos)
        return outputs, targets 
    
    
    def inference(self, dataloader, include_deep_features=False):
        
        return self._inference(dataloader, include_deep_features)
        


    def evaluate(self, dataloader, metric_fn=None):
        """
        Evalúa el modelo en un conjunto de datos.
        """
        # Determina la función de métrica
        metric_fn = metric_fn if metric_fn is not None else self.loss_function 
        
        # Obtiene todas las predicciones y valores verdaderos
        all_predicted, all_targets = self._inference(dataloader)
        
        # Calcula el valor de la métrica y lo devuelve
        metric_value = metric_fn(all_predicted, all_targets)
        return metric_value


#-------------------------------------------------------------------------------------------------------------

class ResNeXtRegressor_ICP(ResNeXtRegressor):
    
    def __init__(self, confidence=0.9, use_metadata=False, meta_input_size=0):
        """
        Inicializa el regresor ResNeXt con ICP (Inductive Conformal Prediction).
        """
        super().__init__(use_metadata=use_metadata, meta_input_size=meta_input_size)
        self.alpha = 1-confidence
        self.q_hat = None 
        
    
    def save_checkpoint(self, save_model_path):
        """
        Guarda el estado del modelo en un archivo checkpoint.
        """
        checkpoint = {
            'pred_model_type': 'ICP',
            'use_metadata': self.use_metadata,
            'torch_state_dict': self.state_dict(),
            'alpha': self.alpha,
            'q_hat': self.q_hat
        }
        torch.save(checkpoint, save_model_path)
    
    
    def load_checkpoint(self, checkpoint):
        """
        Carga el estado del modelo desde un checkpoint
        """
        self.load_state_dict(checkpoint['torch_state_dict'])
        self.alpha = checkpoint['alpha'] if 'alpha' in checkpoint else None
        self.q_hat = checkpoint['q_hat'] if 'q_hat' in checkpoint else None


    def evaluate(self, dataloader, metric_fn=None):
        """
        Evalúa el modelo en un conjunto de datos.
        """
        # Determina la función de métrica
        metric_fn = metric_fn if metric_fn is not None else self.loss_function 
        
        # Obtiene todas las predicciones y valores verdaderos
        all_predicted, all_targets = self._inference(dataloader)
        
        # Calcula el valor de la métrica y lo devuelve
        metric_value = metric_fn(all_predicted, all_targets)
        return metric_value
    
    
    def calibrate(self, calib_loader):
        
        # Obtiene predicciones y valores verdaderos del conjunto de calibración
        calib_pred_values, calib_true_values = self._inference(calib_loader)
        
        # Calcula el nivel de cuantificación ajustado basado en el tamaño del conjunto de calibración y alpha
        n = len(calib_true_values)
        q_level = math.ceil((1.0 - self.alpha) * (n + 1.0)) / n
        
        # Calcula las puntuaciones de no conformidad como valores absolutos de los errores
        nonconformity_scores = torch.abs(calib_true_values-calib_pred_values)
        
        # Calcula el umbral de no conformidad como el cuantil empírico de las puntuaciones de no conformidad
        self.q_hat = torch.quantile(nonconformity_scores, q_level, interpolation='higher')
    
    
    def inference(self, dataloader):
        
        if self.q_hat is None:
            raise ValueError("Modelo no calibrado. Parámetro 'q_hat' no determinado.")
        
        # Obtiene predicciones puntuales y valores reales
        point_pred_values, true_values = self._inference(dataloader)
        
        # Calcula las predicciones interválicas conformales
        lower_pred_values = point_pred_values - self.q_hat
        upper_pred_values = point_pred_values + self.q_hat
        
        # Devuelve las predicciones puntuales, interválicas y valores reales
        return point_pred_values, lower_pred_values, upper_pred_values, true_values
    

#-------------------------------------------------------------------------------------------------------------

class ResNeXtRegressor_QR(ResNeXtRegressor):
    
    def __init__(self, confidence=0.9, use_metadata=False, meta_input_size=0):
        """
        Inicializa el regresor ResNeXt con QR (Quantile Regression).
        """
        super().__init__(use_metadata=use_metadata, meta_input_size=meta_input_size)
        self.alpha = 1-confidence
        self.quantiles = [0.5, self.alpha/2, 1-self.alpha]
        self.loss_function = PinballLoss(self.quantiles)
        
        if self.use_metadata:
            self.classifier = ClassifierResNeXt(
                input_size = 4096+self.embedding.get_last_layer_size(), 
                output_size = len(self.quantiles)
            )
        
        else:
            # Clasificador
            self.classifier = ClassifierResNeXt(input_size = 4096)
        
    
    def save_checkpoint(self, save_model_path):
        """
        Guarda el estado del modelo en un archivo checkpoint.
        """
        checkpoint = {
            'pred_model_type': 'QR',
            'use_metadata': self.use_metadata,
            'torch_state_dict': self.state_dict(),
            'alpha': self.alpha,
            'quantiles': self.quantiles
        }
        torch.save(checkpoint, save_model_path)
        
    
    def load_checkpoint(self, checkpoint):
        """
        Carga el estado del modelo desde un checkpoint
        """
        self.load_state_dict(checkpoint['torch_state_dict'])
        self.alpha = checkpoint['alpha'] if 'alpha' in checkpoint else None
        self.quantiles = checkpoint['quantiles'] if 'quantiles' in checkpoint else None


    def evaluate(self, dataloader, metric_fn=None):
        """
        Evalúa el modelo en un conjunto de datos.
        """
        # Determina la función de métrica
        metric_fn = metric_fn if metric_fn is not None else self.loss_function 
        
        # Obtiene todas las predicciones y valores verdaderos
        all_predicted, all_targets = self._inference(dataloader)
        
        # Calcula el valor de la métrica y lo devuelve
        metric_value = metric_fn(all_predicted, all_targets)
        return metric_value
    
    
    def inference(self, dataloader):
        
        # Obtiene predicciones y valores reales
        outputs, targets = self._inference(dataloader)
        
        # Obtiene las predicciones interválicas y valores reales
        point_pred_values = outputs[:,0]
        lower_pred_values = outputs[:,1]
        upper_pred_values = outputs[:,2]
        true_values = targets
        
        # Asegurar que lower <= point <= upper
        lower_pred_values = np.minimum(lower_pred_values, point_pred_values)
        upper_pred_values = np.maximum(upper_pred_values, point_pred_values)
        
        # Devuelve las predicciones puntuales, interválicas y valores reales
        return point_pred_values, lower_pred_values, upper_pred_values, true_values
    

#-------------------------------------------------------------------------------------------------------------

class ResNeXtRegressor_CQR(ResNeXtRegressor_QR):
    
    def __init__(self, confidence=0.9, use_metadata=False, meta_input_size=0):
        """
        Inicializa el regresor ResNeXt con CQR (Conformalized Quantile Regression).
        """
        super().__init__(confidence, use_metadata=use_metadata, meta_input_size=meta_input_size)
        self.q_hat_lower = None 
        self.q_hat_upper = None
        
    
    def save_checkpoint(self, save_model_path):
        """
        Guarda el estado del modelo en un archivo checkpoint.
        """
        checkpoint = {
            'pred_model_type': 'QR',
            'use_metadata': self.use_metadata,
            'torch_state_dict': self.state_dict(),
            'alpha': self.alpha,
            'quantiles': self.quantiles,
            'q_hat_lower': self.q_hat_lower,
            'q_hat_upper': self.q_hat_upper
        }
        torch.save(checkpoint, save_model_path)
        
    
    def load_checkpoint(self, checkpoint):
        """
        Carga el estado del modelo desde un checkpoint
        """
        self.load_state_dict(checkpoint['torch_state_dict'])
        self.alpha = checkpoint['alpha'] if 'alpha' in checkpoint else None
        self.quantiles = checkpoint['quantiles'] if 'quantiles' in checkpoint else None
        self.q_hat_lower = checkpoint['q_hat_lower'] if 'q_hat_lower' in checkpoint else None
        self.q_hat_upper = checkpoint['q_hat_upper'] if 'q_hat_upper' in checkpoint else None


    def evaluate(self, dataloader, metric_fn=None):
        """
        Evalúa el modelo en un conjunto de datos.
        """
        # Determina la función de métrica
        metric_fn = metric_fn if metric_fn is not None else self.loss_function 
        
        # Obtiene todas las predicciones y valores verdaderos
        all_predicted, all_targets = self._inference(dataloader)
        
        # Calcula el valor de la métrica y lo devuelve
        metric_value = metric_fn(all_predicted, all_targets)
        return metric_value
    
    
    def calibrate(self, calib_loader):
        
        # Obtiene predicciones y valores verdaderos del conjunto de calibración
        _, calib_pred_lower_bound, calib_pred_upper_bound, calib_true_values = \
            super().inference(calib_loader)
        
        # Calcula el nivel de cuantificación ajustado basado en el tamaño del conjunto de calibración y alpha
        n = len(calib_true_values)
        q_level = math.ceil((1.0 - (self.alpha / 2.0)) * (n + 1.0)) / n

        # Calcula las puntuaciones de no conformidad para el límite inferior (diferencia entre predicción 
        # inferior y valor real) y para el límite superior (diferencia entre valor real y predicción superior)
        calib_scores_lower_bound = calib_pred_lower_bound - calib_true_values 
        calib_scores_upper_bound = calib_true_values - calib_pred_upper_bound
        
        # Calcula los umbrales de no conformidad como los cuantiles empíricos de las puntuaciones de no 
        # conformidad
        self.q_hat_lower = torch.quantile(calib_scores_lower_bound, q_level, interpolation='higher')
        self.q_hat_upper = torch.quantile(calib_scores_upper_bound, q_level, interpolation='higher')
    
    
    def inference(self, dataloader):
        
        if self.q_hat_lower is None:
            raise ValueError("Modelo no calibrado. Parámetro 'q_hat_lower' no determinado.")
        
        # Obtiene predicciones puntuales e interválicas y valores reales
        point_pred_values, lower_pred_values, upper_pred_values, true_values = super().inference(dataloader)
        
        # Calcula las predicciones interválicas conformales
        lower_pred_values -= self.q_hat_lower
        upper_pred_values += self.q_hat_upper
        
        # Devuelve las predicciones puntuales, interválicas y valores reales
        return point_pred_values, lower_pred_values, upper_pred_values, true_values


#-------------------------------------------------------------------------------------------------------------

class ResNeXtRegressor_CRF(ResNeXtRegressor):

    def __init__(self, confidence=0.9, use_metadata=False, meta_input_size=0):
        """
        Inicializa el regresor ResNeXt con CRF (Conformalized Residual Fitting).
        """
        super().__init__()
        self.alpha = 1-confidence
        self.q_hat_lower = None
        self.q_hat_upper = None 
        self.sigma_model = None
        
    
    def save_checkpoint(self, save_model_path):
        """
        Guarda el estado del modelo en un archivo checkpoint.
        """
        checkpoint = {
            'pred_model_type': 'CRF',
            'use_metadata': self.use_metadata,
            'torch_state_dict': self.state_dict(),
            'alpha': self.alpha,
            'q_hat_lower': self.q_hat_lower,
            'q_hat_upper': self.q_hat_upper,
            'sigma_model': self.sigma_model
        }
        torch.save(checkpoint, save_model_path)


    def load_checkpoint(self, checkpoint):
        """
        Carga el estado del modelo desde un checkpoint
        """
        self.load_state_dict(checkpoint['torch_state_dict'])
        self.alpha = checkpoint['alpha'] if 'alpha' in checkpoint else None
        self.q_hat_lower = checkpoint['q_hat_lower'] if 'q_hat_lower' in checkpoint else None
        self.q_hat_upper = checkpoint['q_hat_upper'] if 'q_hat_upper' in checkpoint else None
        self.sigma_model = checkpoint['sigma_model'] if 'sigma_model' in checkpoint else None


    def evaluate(self, dataloader, metric_fn=None):
        
        # Determina la función de métrica
        metric_fn = metric_fn if metric_fn is not None else self.loss_function 
        
        # Obtiene todas las predicciones y valores verdaderos
        all_predicted, all_targets = self._inference(dataloader)
        
        # Calcula el valor de la métrica y lo devuelve
        metric_value = metric_fn(all_predicted, all_targets)
        return metric_value


    def _train_sigma_model(self, res_loader, n_estimators=100):
        
        #
        res_pred_values, res_true_values, res_deep_features,  = \
            self._inference(res_loader, include_deep_features=True)
        
        #
        # features = res_deep_features.numpy()
        # errors = torch.abs(res_true_values - res_pred_values).numpy()
        features = res_deep_features
        errors = torch.abs(res_true_values - res_pred_values)

        #
        self.sigma_model = RandomForestRegressor(
            n_estimators=n_estimators, 
            random_state=42,
            n_jobs=-1
        )
        
        #
        self.sigma_model.fit(features, errors)


    def calibrate(self, calib_loader, res_loader, n_estimators=100):
        
        #
        self._train_sigma_model(res_loader, n_estimators)
        
        # Obtiene predicciones y valores verdaderos del conjunto de calibración
        calib_pred_values, calib_true_values, calib_deep_features = \
            self._inference(calib_loader, include_deep_features=True)
            
        # Calcula el nivel de cuantificación ajustado basado en el tamaño del conjunto de calibración y alpha
        n = len(calib_true_values)
        q_level = math.ceil((1.0 - (self.alpha / 2.0)) * (n + 1.0)) / n
        
        #
        sigma_hat_calib = self.sigma_model.predict(calib_deep_features.numpy())
        sigma_hat_calib = np.clip(sigma_hat_calib, 1e-6, None)  # evita divisiones por cero
        
        # Calcula las puntuaciones de no conformidad como valores absolutos de los errores entre la medida de 
        # dispersión predicha
        nonconformity_scores_upper = (calib_true_values - calib_pred_values) / sigma_hat_calib
        nonconformity_scores_lower = -nonconformity_scores_upper
        
        # Calcula el umbral de no conformidad como el cuantil empírico de las puntuaciones de no conformidad
        self.q_hat_upper = torch.quantile(nonconformity_scores_upper, q_level, interpolation='higher')
        self.q_hat_lower = torch.quantile(nonconformity_scores_lower, q_level, interpolation='higher')


    def inference(self, dataloader):
        
        if self.q_hat_lower is None and self.q_hat_upper is None:
            raise ValueError("Modelo no calibrado.")
        
        # Obtiene predicciones, valores verdaderos y características profundas del conjunto de calibración
        point_pred_values, true_values, deep_features = \
            self._inference(dataloader, include_deep_features=True)   
        
        #
        sigma_hat = self.sigma_model.predict(deep_features.numpy())
        sigma_hat = torch.from_numpy(sigma_hat)
        
        #
        upper_pred_values = point_pred_values + self.q_hat_upper * sigma_hat
        lower_pred_values = point_pred_values - self.q_hat_lower * sigma_hat
        
        #
        return point_pred_values, lower_pred_values, upper_pred_values, true_values


#-------------------------------------------------------------------------------------------------------------


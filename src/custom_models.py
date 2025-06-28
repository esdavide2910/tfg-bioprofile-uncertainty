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

# Clase de pérdida para la regresión cuantílica
class SquaredPinballLoss(nn.Module):
    
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
            losses.append(torch.where(errors >= 0, q * errors**2, (1 - q) * errors**2).unsqueeze(1))
            
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
    
    def __init__(self, num_targets=1):

        super(ClassifierResNeXt, self).__init__()  
        
        self.num_targets = num_targets

        self.fc1 = nn.Sequential(
            nn.BatchNorm1d(4096),  # 2048 (avg) + 2048 (max)
            nn.Dropout(p = 0.5),
            nn.Linear(4096, 512),
            nn.ReLU(inplace=True)
        )

        self.fc2 = nn.Sequential(
            nn.BatchNorm1d(512),
            nn.Dropout(p = 0.5),
            nn.Linear(512, num_targets) 
        )
        
        
    def forward(self, x):
        
        x = self.fc1(x)
        x = self.fc2(x)
    
        # nn.Linear siempre devuelve una salida 2D
        # Aplanamos la salida si solo hay una variable target
        return x.squeeze(-1) if self.num_targets==1 else x

#-------------------------------------------------------------------------------------------------------------

class ResNeXtRegressor(nn.Module):
    
    def __init__(self, *args):
        
        super().__init__()
        
        self.feature_extractor = FeatureExtractorResNeXt()
        self.pool_avg = nn.AdaptiveAvgPool2d((1,1))
        self.pool_max = nn.AdaptiveMaxPool2d((1,1))
        self.flatten = nn.Flatten()
        self.classifier = ClassifierResNeXt()
        
        self.loss_function = nn.MSELoss()
        
    
    def save_checkpoint(self, save_model_path):
        
        checkpoint = {
            'pred_model_type': 'base',
            'torch_state_dict': self.state_dict()
        }
        torch.save(checkpoint, save_model_path)


    def load_checkpoint(self, checkpoint):
        
        self.load_state_dict(checkpoint['torch_state_dict'])
        
    
    def forward(self, x, return_deep_features=False):
        
        # Extracción de características
        x = self.feature_extractor(x)
        avg = self.pool_avg(x)
        max = self.pool_max(x)
        x = torch.cat([avg, max], dim=1) 
        x = self.flatten(x)
        
        # Capa FC inicial
        deep_features = self.classifier.fc1(x)  
    
        # Capa FC final 
        outputs = self.classifier.fc2(deep_features)
        outputs = outputs.squeeze(-1) if outputs.dim() > 1 and outputs.shape[-1] == 1 else outputs
        
        # Retorno condicional
        return (outputs, deep_features) if return_deep_features else outputs


    def get_layer_groups(self):
        
        layer_groups = []
        
        layer_groups.append(list(self.feature_extractor.conv1.parameters()))
        layer_groups.append(list(self.feature_extractor.conv2.parameters()))
        layer_groups.append(list(self.feature_extractor.conv3.parameters()))
        layer_groups.append(list(self.feature_extractor.conv4.parameters()))
        layer_groups.append(list(self.feature_extractor.conv5.parameters()))
        layer_groups.append(list(self.classifier.fc1.parameters()))
        layer_groups.append(list(self.classifier.fc2.parameters()))

        return layer_groups


    def classifier_parameters(self):
        return self.classifier.parameters()


    def feature_extractor_parameters(self):
        return self.feature_extractor.parameters()


    def train_epoch(self, dataloader, optimizer, scheduler=None, loss_fn=None):
        
        # Determinamos la función de pérdida
        loss_fn = loss_fn if loss_fn is not None else self.loss_function 
        
        # Pone la red en modo entrenamiento 
        self.train()
        
        # Inicializa la pérdida acumulada para esta época
        epoch_loss = 0
        
        # Itera sobre todos los lotes de datos del DataLoader
        for inputs, targets in dataloader:
            
            # Obtiene las imágenes de entrenamiento y sus valores objetivo
            inputs, targets = inputs.to('cuda'), targets.to('cuda')

            # Limpia los gradientes de la iteración anterior
            optimizer.zero_grad()           
            
            # Pasa las imágenes de entrada a través de la red (propagación hacia adelante)
            outputs = self.forward(inputs)       
            
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
                
                # Verifica si el batch contiene (inputs, targets) o solo inputs
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    inputs, targets = batch
                    has_targets = True
                    all_targets.append(targets.cpu())
                else:
                    inputs = batch # Solo inputs, sin targets
                    
                # Mueve inputs a la gráfica
                inputs = inputs.to('cuda')

                # Ejecuta el modelo y recolecta resultados
                if include_deep_features:
                    outputs, deep_features = self.forward(inputs, return_deep_features=True)
                    all_deep_features.append(deep_features.cpu())
                else:
                    outputs = self.forward(inputs)
                    
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

class ResNeXtRegressor_QR(ResNeXtRegressor):
    
    def __init__(self, region_size=0.9):
        """
        Inicializa el regresor ResNeXt con QR (Quantile Regression).
        """
        super().__init__()
        self.alpha = 1-region_size
        self.quantiles = [0.5, self.alpha/2, 1-self.alpha]
        self.classifier = ClassifierResNeXt(len(self.quantiles))
        self.loss_function = PinballLoss(self.quantiles)
        
    
    def save_checkpoint(self, save_model_path):
        """
        Guarda el estado del modelo en un archivo checkpoint.
        """
        checkpoint = {
            'pred_model_type': 'QR',
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
        self.alpha = checkpoint['alpha']
        self.quantiles = checkpoint['quantiles']


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
        
        # Devuelve las predicciones puntuales, interválicas y valores reales
        return point_pred_values, lower_pred_values, upper_pred_values, true_values


#-------------------------------------------------------------------------------------------------------------

class ResNeXtRegressor_ICP(ResNeXtRegressor):
    
    def __init__(self, confidence=0.9):
        """
        Inicializa el regresor ResNeXt con ICP (Inductive Conformal Prediction).
        """
        super().__init__()
        self.alpha = 1-confidence
        self.q_hat = None 
        
    
    def save_checkpoint(self, save_model_path):
        """
        Guarda el estado del modelo en un archivo checkpoint.
        """
        checkpoint = {
            'pred_model_type': 'ICP',
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
        self.alpha = checkpoint['alpha']
        self.q_hat = checkpoint['q_hat']


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

class ResNeXtRegressor_CQR(ResNeXtRegressor_QR):
    
    def __init__(self, confidence=0.9):
        """
        Inicializa el regresor ResNeXt con CQR (Conformalized Quantile Regression).
        """
        super().__init__(confidence)
        self.q_hat_lower = None 
        self.q_hat_upper = None
        
    
    def save_checkpoint(self, save_model_path):
        """
        Guarda el estado del modelo en un archivo checkpoint.
        """
        checkpoint = {
            'pred_model_type': 'QR',
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
        self.alpha = checkpoint['alpha']
        self.quantiles = checkpoint['quantiles']
        self.q_hat_lower = checkpoint['q_hat_lower']
        self.q_hat_upper = checkpoint['q_hat_upper']


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

    def __init__(self, confidence=0.9):
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
        self.alpha = checkpoint['alpha']
        try:
            self.q_hat_lower = checkpoint['q_hat_lower']
            self.q_hat_upper = checkpoint['q_hat_upper']
            self.sigma_model = checkpoint['sigma_model']
        except:
            pass


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


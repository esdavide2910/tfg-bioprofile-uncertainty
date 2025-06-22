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
        


    def _forward_deep_features(self, x):
        
        x = self.feature_extractor(x)
        avg = self.pool_avg(x)
        max = self.pool_max(x)
        x = torch.cat([avg, max], dim=1) 
        x = self.flatten(x)
        x = self.classifier.fc1(x)
        
        return x 
    
    
    def _forward_last_layer(self, x):
        
        return self.classifier.fc2(x)


    def forward(self, x):
        
        x = self._forward_deep_features(x)
        x = self._forward_last_layer(x)
        
        return x.squeeze(-1)


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
                    deep_features = self._forward_deep_features(inputs)
                    all_deep_features.append(deep_features)
                    outputs = self._forward_last_layer(inputs)
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
        
        # Determinamos la función de métrica
        metric_fn = metric_fn if metric_fn is not None else self.loss_function 
        
        #
        all_predicted, all_targets = self.inference(dataloader)
        
        #
        metric_value = metric_fn(all_predicted, all_targets)
        return metric_value
        

#-------------------------------------------------------------------------------------------------------------

class ResNeXtRegressor_QR(ResNeXtRegressor):
    
    def __init__(self, region_size=0.9):
        
        super().__init__()
        
        self.alpha = 1-region_size
        self.quantiles = [0.5, self.alpha/2, 1-self.alpha]
        self.classifier = ClassifierResNeXt(len(self.quantiles))
        self.loss_function = SquaredPinballLoss(self.quantiles)
        
    
    def save_checkpoint(self, save_model_path):
        
        checkpoint = {
            'pred_model_type': 'QR',
            'torch_state_dict': self.state_dict(),
            'alpha': self.alpha,
            'quantiles': self.quantiles
        }
        torch.save(checkpoint, save_model_path)
        
    
    def load_checkpoint(self, checkpoint):
        
        self.load_state_dict(checkpoint['torch_state_dict'])
        self.alpha = checkpoint['alpha']
        self.quantiles = checkpoint['quantiles']


    def forward(self, x):
        
        x = self.feature_extractor(x)
        avg = self.pool_avg(x)
        max = self.pool_max(x)
        x = torch.cat([avg, max], dim=1) 
        x = self.flatten(x)
        x = self.classifier.fc1(x)
        x = self.classifier.fc2(x)
        
        return x
    
    
    # def _inference(self, dataloader):
    
    
    def inference(self, dataloader):
        
        outputs, targets = self._inference(dataloader)
        
        point_pred_values = outputs[:,0]
        lower_pred_values = outputs[:,1]
        upper_pred_values = outputs[:,2]
        true_values = targets
        
        return point_pred_values, lower_pred_values, upper_pred_values, true_values


    def evaluate(self, dataloader, metric_fn=None):
        
        # Determinamos la función de métrica
        metric_fn = metric_fn if metric_fn is not None else self.loss_function 
        
        #
        all_predicted, all_targets =  self._inference(dataloader)
        
        #
        metric_value = metric_fn(all_predicted, all_targets)
        return metric_value


#-------------------------------------------------------------------------------------------------------------

class ResNeXtRegressor_ICP(ResNeXtRegressor):
    
    def __init__(self, confidence=0.9):
        
        super().__init__()
        self.alpha = 1-confidence
        self.q_hat = None 
        
    
    def save_checkpoint(self, save_model_path):
        
        checkpoint = {
            'pred_model_type': 'ICP',
            'torch_state_dict': self.state_dict(),
            'alpha': self.alpha,
            'q_hat': self.q_hat
        }
        torch.save(checkpoint, save_model_path)
        
    
    def load_checkpoint(self, checkpoint):
        
        self.load_state_dict(checkpoint['torch_state_dict'])
        self.alpha = checkpoint['alpha']
        self.q_hat = checkpoint['q_hat']
        
        
    def calibrate(self, calib_loader):
        
        # Obtener predicciones y valores verdaderos del conjunto de calibración
        calib_pred_values, calib_true_values = self._inference(calib_loader)
        
        # Calcula el nivel de cuantificación ajustado basado en el tamaño del conjunto de calibración y alpha
        n = len(calib_true_values)
        q_level = math.ceil((1.0 - self.alpha) * (n + 1.0)) / n
        
        # Calcula las puntuaciones de calibración como valores absolutos de los errores
        calib_scores = torch.abs(calib_true_values-calib_pred_values)
        
        # Calcula el cuantil q_hat usado para ajustar el intervalo predictivo
        self.q_hat = torch.quantile(calib_scores, q_level, interpolation='higher')
    
    
    
    def inference(self, dataloader):
        
        if self.q_hat is None:
            raise ValueError("Modelo no calibrado. Parámetro 'q_hat' no determinado.")
        
        outputs, targets = self._inference(dataloader)
        
        point_pred_values = outputs
        lower_pred_values = outputs - self.q_hat
        upper_pred_values = outputs + self.q_hat
        true_values = targets
        
        return point_pred_values, lower_pred_values, upper_pred_values, true_values


    def evaluate(self, dataloader, metric_fn=None):
        
        # Determinamos la función de métrica
        metric_fn = metric_fn if metric_fn is not None else self.loss_function 
        
        #
        all_predicted, all_targets = self._inference(dataloader)
        
        #
        metric_value = metric_fn(all_predicted, all_targets)
        return metric_value
    

#-------------------------------------------------------------------------------------------------------------

class ResNeXtRegressor_CQR(ResNeXtRegressor_QR):
    
    def __init__(self, confidence=0.9):
        
        super().__init__(confidence)
        self.q_hat_lower = None 
        self.q_hat_upper = None
        
    
    def save_checkpoint(self, save_model_path):
        
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
        
        self.load_state_dict(checkpoint['torch_state_dict'])
        self.alpha = checkpoint['alpha']
        self.quantiles = checkpoint['quantiles']
        self.q_hat_lower = checkpoint['q_hat_lower']
        self.q_hat_upper = checkpoint['q_hat_upper']
    
    
    def calibrate(self, calib_loader):
        
        # Obtener predicciones y valores verdaderos del conjunto de calibración
        _, calib_pred_lower_bound, calib_pred_upper_bound, calib_true_values = \
            self._inference(calib_loader)
        
        # Calcula el nivel de cuantificación ajustado basado en el tamaño del conjunto de calibración y alpha
        n = len(calib_true_values)
        q_level = math.ceil((1.0 - (self.alpha / 2.0)) * (n + 1.0)) / n

        # Calcula las puntuaciones para el límite inferior (diferencia entre predicción inferior y valor real)
        # y para el límite superior (diferencia entre valor real y predicción superior)
        calib_scores_lower_bound = calib_pred_lower_bound - calib_true_values
        calib_scores_upper_bound = calib_true_values - calib_pred_upper_bound
        
        # Calcula los cuantiles q_hat para ambos límites del intervalo predictivo
        self.q_hat_lower = torch.quantile(calib_scores_lower_bound, q_level, interpolation='higher')
        self.q_hat_upper = torch.quantile(calib_scores_upper_bound, q_level, interpolation='higher')
    
    
    def inference(self, dataloader):
        
        if self.q_hat_lower is None:
            raise ValueError("Modelo no calibrado. Parámetro 'q_hat_lower' no determinado.")
        
        point_pred_values, lower_pred_values, upper_pred_values, true_values = super().inference(dataloader)
        
        lower_pred_values -= self.q_hat_lower
        upper_pred_values += self.q_hat_upper
        
        return point_pred_values, lower_pred_values, upper_pred_values, true_values


    def evaluate(self, dataloader, metric_fn=None):
        
        # Determinamos la función de métrica
        metric_fn = metric_fn if metric_fn is not None else self.loss_function 
        
        #
        all_predicted, all_targets = self._inference(dataloader)
        
        #
        metric_value = metric_fn(all_predicted, all_targets)
        return metric_value


#-------------------------------------------------------------------------------------------------------------

class ResNeXtRegressor_CRF(ResNeXtRegressor):
    
    def __init__(self, confidence=0.9):
        
        super().__init__()
        self.alpha = 1-confidence
        self.q_hat = None 
        self.sigma_model = None
        
    
    def save_checkpoint(self, save_model_path):
        
        checkpoint = {
            'pred_model_type': 'CRF',
            'torch_state_dict': self.state_dict(),
            'alpha': self.alpha,
            'q_hat': self.q_hat,
            'sigma_model': self.sigma_model
        }
        torch.save(checkpoint, save_model_path)
        
    
    def load_checkpoint(self, checkpoint):
        
        self.load_state_dict(checkpoint['torch_state_dict'])
        self.alpha = checkpoint['alpha']
        self.q_hat = checkpoint['q_hat']
        self.sigma_model = checkpoint['sigma_model']

        
    def calibrate(self, calib_loader, res_loader, n_estimators=100):
        
        #
        res_pred_values, res_true_values, res_deep_features,  = \
            self._inference(res_loader, include_deep_features=True)
        
        #
        X = res_deep_features.numpy()
        y = np.abs(res_true_values - res_pred_values)

        #
        self.sigma_model = RandomForestRegressor(
            n_estimators=n_estimators, 
            random_state=42,
            n_jobs=-1
        )
        self.sigma_model.fit(X,y)
        
        # Obtener predicciones y valores verdaderos del conjunto de calibración
        calib_pred_values, calib_true_values, calib_deep_features = \
            self._inference(calib_loader, include_deep_features=True)
            
        # Calcula el nivel de cuantificación ajustado basado en el tamaño del conjunto de calibración y alpha
        n = len(calib_true_values)
        q_level = math.ceil((1.0 - (self.alpha / 2.0)) * (n + 1.0)) / n
        
        #
        sigma_hat_calib = self.sigma_model.predict(calib_deep_features.numpy())
        sigma_hat_calib = np.clip(sigma_hat_calib, 1e-6, None)  # evitar divisiones por cero
        
        # Calcula las puntuaciones de calibración como valores absolutos de los errores entre la medida de 
        # dispersión predicha
        sigma_hat_calib = torch.from_numpy(sigma_hat_calib)
        calib_scores = torch.abs(calib_true_values - calib_pred_values) / sigma_hat_calib
        
        # Calcula el cuantil q_hat usado para ajustar el intervalo predictivo
        self.q_hat = torch.quantile(calib_scores, q_level, interpolation='higher')
    
    
    def inference(self, dataloader):
        
        if self.q_hat is None:
            raise ValueError("Modelo no calibrado. Parámetro 'q_hat' no determinado.")
        
        pred_values, true_values, deep_features = \
            self._inference(dataloader, include_deep_features=True)   
        
        sigma_hat_test = self.sigma_model.predict(deep_features.numpy())
        sigma_hat_test = torch.from_numpy(sigma_hat_test)
        
        point_pred_values = pred_values
        lower_pred_values = pred_values - self.q_hat * sigma_hat_test
        upper_pred_values = pred_values + self.q_hat * sigma_hat_test
        
        return point_pred_values, lower_pred_values, upper_pred_values, true_values


    def evaluate(self, dataloader, metric_fn=None):
        
        # Determinamos la función de métrica
        metric_fn = metric_fn if metric_fn is not None else self.loss_function 
        
        #
        all_predicted, all_targets = self._inference(dataloader)
        
        #
        metric_value = metric_fn(all_predicted, all_targets)
        return metric_value
    
    
#-------------------------------------------------------------------------------------------------------------


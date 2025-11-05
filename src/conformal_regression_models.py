#-------------------------------------------------------------------------------------------------------------
# BIBLIOTECAS ------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------
import torch
import torchvision 
import torch.nn as nn
import math
import copy

#-------------------------------------------------------------------------------------------------------------


#-------------------------------------------------------------------------------------------------------------
# COMPONENTES BÁSICOS ----------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------

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

        super().__init__()  
        
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
        # Aplana la salida si solo hay una variable target
        return x.squeeze(-1) if x.dim() > 1 and x.shape[-1] == 1 else x

#-------------------------------------------------------------------------------------------------------------


#-------------------------------------------------------------------------------------------------------------
# MODELOS COMPLETOS ------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------

class ResNeXtRegressor(nn.Module):
    
    PRED_METHOD = 'base'
    
    def __init__(self, **kwargs):
        
        # Inicializa la clase padre
        super().__init__()
        
        # Define las componentes de la red: 
        # 1) Extractor de características, pooling y flattening 
        self.feature_extractor = FeatureExtractorResNeXt()
        self.pool_avg = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()
        
        # 2) Classifier
        input_size = 2048 # características aplanadas 
        self.num_outputs = 1
        self.classifier = ClassifierResNeXt(input_size, output_size=self.num_outputs)
        
        # Define la función de pérdida
        self.loss_function = nn.MSELoss()
    
    
    def save_checkpoint(self, save_model_path):
        """Guarda el estado actual de un modelo y parámetros de calibración en un archivo"""
        checkpoint = {
            'pred_method': self.PRED_METHOD,
            'torch_state_dict': self.state_dict()
        }
        torch.save(checkpoint, save_model_path)


    def load_checkpoint(self, checkpoint):
        """Carga el estado del modelo desde un checkpoint"""
        weight_key = 'classifier.fc2.2.weight'
        bias_key = 'classifier.fc2.2.bias'
        
        if weight_key in checkpoint['torch_state_dict']:
            out_features = checkpoint['torch_state_dict'][weight_key].shape[0]
            
            if out_features != self.num_outputs:
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
        outputs =  self.classifier(x)
        
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
        layer_groups.append(list(self.classifier.fc1.parameters()))
        layer_groups.append(list(self.classifier.fc2.parameters()))
        return layer_groups


    def train_epoch(self, dataloader, optimizer, scheduler=None, loss_fn=None):
        """Entrena el modelo por una época completa"""
        
        # Determina la función de pérdida
        loss_fn = loss_fn if loss_fn is not None else self.loss_function 
        
        # Pone la red en modo entrenamiento 
        self.train()
        
        # Inicializa la pérdida acumulada para esta época
        epoch_loss = 0
        
        # Itera sobre todos los batches del dataloader
        for images, targets in dataloader:
            
            # Obtiene las imágenes de entrenamiento y sus valores objetivo
            images, targets = images.to('cuda'), targets.to('cuda')
            
            # Limpia los gradientes de la iteración anterior
            optimizer.zero_grad()
            
            # Obtiene los valores predichos del modelo
            outputs = self.forward(images)
            
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
                
                # Verifica si el batch contiene (image, target) o solo (image)
                    
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    images, targets = batch 
                    images = images.to('cuda')
                    has_targets = True
                    all_targets.append(targets.cpu())
                    
                else:
                    images = batch
                    images = images.to('cuda')

                # Ejecuta el modelo y recolecta resultados
                outputs = self.forward(images)
                all_outputs.append(outputs.cpu())
                
        # Concatena los resultados
        outputs = torch.cat(all_outputs)
        targets = torch.cat(all_targets) if has_targets else None

        # Devuelve los valores predichos y valores verdaderos
        return outputs, targets 
    
    
    def inference(self, dataloader, valid_loader):
        
        #
        pred_point_values, true_values = self._inference(dataloader)
        
        #
        valid_pred_point_values, valid_true_values = self._inference(valid_loader)
        valid_mean = torch.mean(valid_true_values - valid_pred_point_values)
        valid_std = torch.std(valid_true_values - valid_pred_point_values)
        
        #
        pred_point_values += valid_mean
        z = 1.96  # para un 95% de cobertura bajo normalidad
        pred_lower_bound = pred_point_values - z * valid_std
        pred_upper_bound = pred_point_values + z * valid_std
        
        return pred_point_values, pred_lower_bound, pred_upper_bound, true_values


    def evaluate(self, dataloader, loss_fn=None):
        """
        Evalúa el modelo en un conjunto de datos.
        """
        # Determina la función de métrica
        loss_fn = loss_fn if loss_fn is not None else self.loss_function 
        
        # Obtiene todas las predicciones y valores verdaderos
        all_predicted, all_targets = self._inference(dataloader)
        
        # Calcula el valor de la métrica y lo devuelve
        metric_value = loss_fn(all_predicted, all_targets)
        return metric_value


#-------------------------------------------------------------------------------------------------------------

class ResNeXtRegressor_ICP(ResNeXtRegressor):
    
    PRED_METHOD = 'ICP'
    
    def __init__(self, confidence=0.95):
        """
        Inicializa el regresor ResNeXt con ICP (Inductive Conformal Prediction).
        """
        # Inicializa la clase padre
        super().__init__()
        
        # Parámetros para la conformal prediction
        self.alpha = 1-confidence
        self.delta = None 
        
    
    def save_checkpoint(self, save_model_path):
        """
        Guarda el estado del modelo en un archivo checkpoint.
        """
        checkpoint = {
            'pred_method': self.PRED_METHOD,
            'torch_state_dict': self.state_dict(),
            'alpha': self.alpha,
            'delta': self.delta
        }
        torch.save(checkpoint, save_model_path)
    
    
    def load_checkpoint(self, checkpoint):
        """
        Carga el estado del modelo desde un checkpoint
        """
        weight_key = 'classifier.fc2.2.weight'
        bias_key = 'classifier.fc2.2.bias'
        
        if weight_key in checkpoint['torch_state_dict']:
            out_features = checkpoint['torch_state_dict'][weight_key].shape[0]
            
            if out_features != self.num_outputs:
                checkpoint['torch_state_dict'].pop(weight_key, None)
                checkpoint['torch_state_dict'].pop(bias_key, None)
        
        self.load_state_dict(checkpoint['torch_state_dict'], strict=False)
        
        if (checkpoint['pred_method'] == self.PRED_METHOD and
            checkpoint['alpha'] == self.alpha and
            'delta' in checkpoint
        ):
            self.delta = checkpoint['delta']


    def evaluate(self, dataloader, loss_fn=None):
        """
        Evalúa el modelo en un conjunto de datos.
        """
        # Determina la función de pérdida
        loss_fn = loss_fn if loss_fn is not None else self.loss_function 
        
        # Obtiene todas las predicciones y valores verdaderos
        all_predicted, all_targets = self._inference(dataloader)
        
        # Calcula el valor de la función de pérdida y lo devuelve
        metric_value = loss_fn(all_predicted, all_targets)
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
        self.delta_middle = torch.quantile(calib_true_values-calib_pred_values, 0.5, interpolation='higher')
        self.delta = torch.quantile(nonconformity_scores, q_level, interpolation='higher')
    
    
    def inference(self, dataloader):
        
        if self.delta is None:
            raise ValueError("Modelo no calibrado")
        
        # Obtiene predicciones puntuales y valores reales
        point_pred_values, true_values = self._inference(dataloader)
        
        # Calcula las predicciones interválicas conformales
        point_pred_values = point_pred_values + self.delta_middle
        lower_pred_values = point_pred_values - self.delta
        upper_pred_values = point_pred_values + self.delta
        
        # Devuelve las predicciones puntuales, interválicas y valores reales
        return point_pred_values, lower_pred_values, upper_pred_values, true_values
    

#-------------------------------------------------------------------------------------------------------------

class ResNeXtRegressor_QR(ResNeXtRegressor):
    
    PRED_METHOD = 'QR'
    
    def __init__(self, confidence=0.95):
        """
        Inicializa el regresor ResNeXt con QR (Quantile Regression).
        """
        # Inicializa la clase padre
        super().__init__()
        
        # Parámetros de Quantile Regression
        self.alpha = 1-confidence
        self.quantiles = [0.5, self.alpha/2, 1-self.alpha]
        
        # Define una nueva cabecera
        input_size = 2048
        output_size = len(self.quantiles)
        self.classifier = ClassifierResNeXt(input_size, output_size)
        
        # Define la función de pérdida
        self.loss_function = PinballLoss(self.quantiles)
        
    
    def save_checkpoint(self, save_model_path):
        """
        Guarda el estado del modelo en un archivo checkpoint.
        """
        checkpoint = {
            'pred_method': self.PRED_METHOD,
            'torch_state_dict': self.state_dict(),
            'alpha': self.alpha,
            'quantiles': self.quantiles
        }
        torch.save(checkpoint, save_model_path)
        
    
    def load_checkpoint(self, checkpoint):
        """
        Carga el estado del modelo desde un checkpoint
        """
        weight_key = 'classifier.fc2.2.weight'
        bias_key = 'classifier.fc2.2.bias'
        
        if weight_key in checkpoint['torch_state_dict']:
            out_features = checkpoint['torch_state_dict'][weight_key].shape[0]
            
            if out_features != self.num_outputs:
                checkpoint['torch_state_dict'].pop(weight_key, None)
                checkpoint['torch_state_dict'].pop(bias_key, None)
        
        self.load_state_dict(checkpoint['torch_state_dict'], strict=False)
        self.quantiles = checkpoint['quantiles']
        self.alpha = checkpoint['alpha']
    
    
    def inference(self, dataloader):
        
        # Obtiene predicciones y valores reales
        pred_values, targets = self._inference(dataloader)
        
        # Obtiene las predicciones interválicas y valores reales
        point_pred_values = pred_values[:,0]
        lower_pred_values = pred_values[:,1]
        upper_pred_values = pred_values[:,2]
        true_values = targets
        
        # Asegurar que lower <= point <= upper
        lower_pred_values = torch.minimum(lower_pred_values, point_pred_values)
        upper_pred_values = torch.maximum(upper_pred_values, point_pred_values)
        
        # Devuelve las predicciones puntuales, interválicas y valores reales
        return point_pred_values, lower_pred_values, upper_pred_values, true_values


#-------------------------------------------------------------------------------------------------------------

class ResNeXtRegressor_CQR(ResNeXtRegressor_QR):
    
    PRED_METHOD = 'CQR'
    
    def __init__(self, confidence=0.95):
        """
        Inicializa el regresor ResNeXt con CQR (Conformalized Quantile Regression).
        """
        # Inicializa la clase padre
        super().__init__(confidence)
        
        # Parámetros para la conformal prediction
        self.delta_lower = None 
        self.delta_upper = None
    
    
    def save_checkpoint(self, save_model_path):
        """
        Guarda el estado del modelo en un archivo checkpoint.
        """
        checkpoint = {
            'pred_method': self.PRED_METHOD,
            'torch_state_dict': self.state_dict(),
            'alpha': self.alpha,
            'quantiles': self.quantiles,
            'delta_lower': self.delta_lower,
            'delta_upper': self.delta_upper
        }
        torch.save(checkpoint, save_model_path)
        
    
    def load_checkpoint(self, checkpoint):
        """
        Carga el estado del modelo desde un checkpoint
        """
        weight_key = 'classifier.fc2.2.weight'
        bias_key = 'classifier.fc2.2.bias'
        
        if weight_key in checkpoint['torch_state_dict']:
            out_features = checkpoint['torch_state_dict'][weight_key].shape[0]
            
            if out_features != self.num_outputs:
                checkpoint['torch_state_dict'].pop(weight_key, None)
                checkpoint['torch_state_dict'].pop(bias_key, None)
                
        self.load_state_dict(checkpoint['torch_state_dict'], strict=False)
        
        if (checkpoint['pred_method'] == self.PRED_METHOD and
            checkpoint['quantiles'] == self.quantiles and
            checkpoint['alpha'] == self.alpha and
            'delta_lower' in checkpoint and
            'delta_upper' in checkpoint
        ):
            self.delta_lower = checkpoint['delta_lower']
            self.delta_upper = checkpoint['delta_upper']
        
        self.quantiles = checkpoint['quantiles']
        self.alpha = checkpoint['alpha']
    
    
    def calibrate(self, calib_loader):
        
        # Obtiene predicciones y valores verdaderos del conjunto de calibración
        calib_pred_middle, calib_pred_lower_bound, calib_pred_upper_bound, calib_true_values = \
            super().inference(calib_loader)
        
        # Calcula el nivel de cuantificación ajustado basado en el tamaño del conjunto de calibración y alpha
        n = len(calib_true_values)
        q_level = math.ceil((1.0 - (self.alpha / 2.0)) * (n + 1.0)) / n

        # Calcula las puntuaciones de no conformidad para el límite inferior (diferencia entre predicción 
        # inferior y valor real) y para el límite superior (diferencia entre valor real y predicción superior)
        calib_scores_lower_bound = calib_pred_lower_bound - calib_true_values 
        calib_scores_middle_bound = calib_true_values - calib_pred_middle
        calib_scores_upper_bound = calib_true_values - calib_pred_upper_bound
        
        # Calcula los umbrales de no conformidad como los cuantiles empíricos de las puntuaciones de no 
        # conformidad
        self.delta_lower = torch.quantile(calib_scores_lower_bound, q_level, interpolation='higher')
        self.delta_middle = torch.quantile(calib_scores_middle_bound, 0.5, interpolation='higher')
        self.delta_upper = torch.quantile(calib_scores_upper_bound, q_level, interpolation='higher')
    
    
    def inference(self, dataloader):
        
        if self.delta_lower is None or self.delta_upper is None:
            raise ValueError("Modelo no calibrado")
        
        # Obtiene predicciones puntuales e interválicas y valores reales
        point_pred_values, lower_pred_values, upper_pred_values, true_values = super().inference(dataloader)
        
        # Calcula las predicciones interválicas conformales
        lower_pred_values -= self.delta_lower
        point_pred_values += self.delta_middle
        upper_pred_values += self.delta_upper
        
        # Devuelve las predicciones puntuales, interválicas y valores reales
        return point_pred_values, lower_pred_values, upper_pred_values, true_values

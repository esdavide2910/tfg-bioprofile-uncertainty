import torch
import torchvision 
import torch.nn as nn
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import math
from conformal_regression_models import *
import copy


class ResNeXtRegressor_SCCP(ResNeXtRegressor):

    def __init__(self, confidence=0.9, use_metadata=False, meta_input_size=0):
        """
        Inicializa el regresor ResNeXt con CRF (Conformalized Residual Fitting).
        """
        super().__init__()
        self.alpha = 1-confidence
        self.q_hat_lower = None
        self.q_hat_upper = None
        
        # 
        input_size = 2048
        if use_metadata:
            input_size += 16
        
        #
        self.sigma_model = ResNeXtRegressor(use_metadata, meta_input_size)
    
    
    
    def save_checkpoint(self, save_model_path):
        """
        Guarda el estado del modelo en un archivo checkpoint.
        """
        checkpoint = {
            'pred_model_type': 'SCCP',
            'use_metadata': self.use_metadata,
            'torch_state_dict': self.state_dict(),
            'alpha': self.alpha,
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
        self.q_hat_lower = checkpoint['q_hat_lower'] if 'q_hat_lower' in checkpoint else None
        self.q_hat_upper = checkpoint['q_hat_upper'] if 'q_hat_upper' in checkpoint else None
        
        
    def _inference_sigma(self, dataloader):
        
        # Pone la red en modo evaluación
        self.sigma_model.eval()
        
        # Inicializa listas 
        all_targets = [] 
        all_outputs = []
        
        # Desactiva el cálculo de gradientes para eficiencia
        with torch.no_grad():
            for batch in dataloader:

                # Verifica si el batch contiene (images, metadata, targets) o solo (images, metadata)
                if isinstance(batch, (list, tuple)) and len(batch) == 3:
                    images, metadata, targets = batch 
                    images, metadata = images.to('cuda'), metadata.to('cuda')

                elif isinstance(batch, (list, tuple)) and len(batch) == 2:
                    images, metadata = batch 
                    images, metadata = images.to('cuda'), metadata.to('cuda')

                else:
                    images = batch
                    images = images.to('cuda')
                    metadata = None

                # Ejecuta el modelo y recolecta resultados
                outputs = self.forward(images, metadata)
                true_error = targets-outputs
                pred_error = self.sigma_model.forward(images, metadata)
                all_outputs.append(pred_error.cpu())
                all_targets.append(true_error.cpu())
        
        #
        outputs = torch.cat(all_outputs)
        targets = torch.cat(all_targets) 

        # Devuelve una tupla (valores predichos, valores verdaderos)
        return outputs, targets 
        


    def _train_epoch_sigma(self, dataloader, optimizer, scheduler=None):
        
        # Determinamos la función de pérdida
        loss_fn = nn.MSELoss()
        
        # Pone la red en modo entrenamiento 
        self.sigma_model.train()
        
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
            #
            error = targets-outputs
            
            #
            sigma = self.sigma_model.forward(images, metadata)
            
            
            # Calcula la pérdida de las predicciones
            loss = loss_fn(sigma, error)
            
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
    
    
    def train_sigma_model(self, train_loader, valid_loader):
        
        # Copia el extractor de características del modelo completo al modelo de estimación del error 
        self.sigma_model.feature_extractor = self.feature_extractor
        
        # Congela los parámetros del extractor de características
        for param in self.sigma_model.feature_extractor.parameters():
            param.requires_grad = False
            
        #
        if self.use_metadata:
            # Lista de grupos de parámetros con diferentes configuraciones
            parameters = [
                {'params': self.sigma_model.classifier.fc2.parameters(), 'lr': 3e-2},
                {'params': self.sigma_model.classifier.fc1.parameters(), 'lr': 2e-2},
                {'params': self.sigma_model.embedding.parameters(), 'lr': 2e-2}
            ]
            max_lrs = [3e-2, 2e-2, 2e-2]
        else:
            parameters = [
                {'params': self.sigma_model.classifier.fc2.parameters(), 'lr': 3e-2},
                {'params': self.sigma_model.classifier.fc1.parameters(), 'lr': 2e-2},
            ]
            max_lrs = [3e-2, 2e-2]
            
        #
        optimizer = torch.optim.AdamW(parameters, weight_decay=5e-4)
        
        # Número máximo de épocas a entrenar
        MAX_EPOCHS_SIGMA = 10
        
        # Número mínimo de épocas a entrenar
        MIN_EPOCHS_SIGMA = 10 
        
        # Número de épocas sin mejora antes de detener el entrenamiento
        PATIENCE = 5
        
        # Inicializa la mejor pérdida de validación como la obtenida en el entrenamiento de la cabecera
        best_valid_loss = float('inf')
        
        #  Contador de épocas sin mejora
        epochs_no_improve = 0
        
        #
        best_model_state = None  
        
        # Crea el scheduler OneCycleLR
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=max_lrs, 
            steps_per_epoch=len(train_loader),
            epochs=MAX_EPOCHS_SIGMA
        )

        for epoch in range(MAX_EPOCHS_SIGMA):

            # 
            sigma_train_loss = self._train_epoch_sigma(train_loader, optimizer, scheduler)

            # 
            sigma_valid_loss = self.sigma_model.evaluate(valid_loader)

            # Imprime los valores de pérdida obtenidos en entrenamiento y validación 
            print(
                f"Epoch {epoch+1:>2} | "+
                f"Train Loss: {sigma_train_loss:>7.3f} | " + 
                f"Validation Loss: {sigma_valid_loss:>7.3f} | " 
            )
            
            # Comprueba si la pérdida en validación ha mejorado
            if sigma_valid_loss < best_valid_loss:
                
                # Actualiza la mejor pérdida en validación obtenida hasta ahora
                best_valid_loss = sigma_valid_loss
                
                # Reinicia el contador de épocas sin mejora si la pérdida ha mejorado
                epochs_no_improve = 0
                
                # Guarda los pesos del modelo actual como los mejores hasta ahora
                best_model_state = copy.deepcopy(self.sigma_model.state_dict())
                
            else:
                # Incrementa el contador si no hay mejora en la pérdida de validación
                epochs_no_improve += 1
                
            if epochs_no_improve >= PATIENCE and (epoch+1) > MIN_EPOCHS_SIGMA: 
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        #
        self.sigma_model.load_state_dict(best_model_state)
        
        print("✅ Entrenamiento de la red sigma completado")
    

    def calibrate(self, calib_loader, res_loader):
        
        #
        calib_pred_sigma, calib_true_sigma = self._inference_sigma(calib_loader)
        
        # Calcula las puntuaciones de no conformidad como 
        nonconformity_scores_upper = calib_true_sigma / calib_pred_sigma
        nonconformity_scores_lower = -nonconformity_scores_upper
        
        # Calcula el nivel de cuantificación ajustado basado en el tamaño del conjunto de calibración y alpha
        n = len(calib_pred_sigma)
        q_level = math.ceil((1.0 - (self.alpha / 2.0)) * (n + 1.0)) / n
        
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

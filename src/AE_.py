
import torch
import numpy as np


class AE_ModelRunner:
    
    PRED_MODEL_TYPE = 'base'
    
    def __init__(self, model, model_path, device=None):
        
        self.model = model
        self.default_model_path = model_path
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    def _verify_checkpoint_type(self, checkpoint):
        """Verifica que el checkpoint sea compatible con el modelo."""
        if checkpoint.get('pred_model_type') != self.PRED_MODEL_TYPE:
            raise ValueError(
                f"Tipo de modelo incompatible: se esperaba '{self.PRED_MODEL_TYPE}', "
                f"pero se encontró '{checkpoint.get('pred_model_type')}'"
            )
    
    
    def save_model(self, model_path=None):
        
        model_path = self.default_model_path if model_path is None else model_path
        
        # Crea el checkpoint con el tipo de modelo y sus pesos
        checkpoint = {
            'pred_model_type': self.PRED_MODEL_TYPE,
            'model_state_dict': self.model.state_dict()
        }
        
        # Guarda el checkpoint 
        torch.save(checkpoint, model_path)

    
    def load_model(self, model_path=None):
        
        model_path = self.default_model_path if model_path is None else model_path
        
        try:
            
            # Carga el checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Verifica que el tipo de modelo coincida
            self._verify_checkpoint_type(checkpoint)
            
            # Carga los pesos del modelo
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
        except Exception as e:
            raise RuntimeError(f"Error al cargar el modelo desde {model_path}: {e}") from e

    
    def train_epoch(self, dataloader, loss_fn, optimizer):
        
        # Pone la red en modo entrenamiento (esto habilita el dropout)
        self.model.train()  
        
        # Inicializa la pérdida acumulada para esta época
        epoch_loss = 0

        # Itera sobre todos los lotes de datos del DataLoader
        for inputs, targets in dataloader:
            
            # Obtiene las imágenes de entrenamiento y sus valores objetivo
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Limpia los gradientes de la iteración anterior
            optimizer.zero_grad()           
            
            # Pasa las imágenes de entrada a través de la red (propagación hacia adelante)
            outputs = self.model(inputs)       
            
            # Calcula la pérdida de las predicciones
            loss = loss_fn(outputs, targets) 
            
            # Realiza la retropropagación para calcular los gradientes (propagación hacia atrás)
            loss.backward()
            
            # Actualiza los parámetros del modelo
            optimizer.step()            
            
            # Actualiza el scheduler de la tasa de aprendizaje (si se proporciona)
            if self.scheduler is not None:
                self.scheduler.step()   
    
            # Acumula la pérdida de este batch
            epoch_loss += loss.item()        
        
        # Calcula la pérdida promedio de la época y la devolvemos
        avg_loss = epoch_loss / len(dataloader)
        return avg_loss
    
    
    def inference(self, dataloader, return_features=False):
        
        # Pone la red en modo evaluación 
        self.model.eval()
        
        # Inicializa listas si son requeridas
        all_targets = [] 
        all_outputs = [] 
        all_features = [] if return_features else None
        
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
                    
                # Mueve inputs al dispositivo correspondiente (GPU o CPU)
                inputs = inputs.to(self.device)

                # Ejecuta el modelo en el modo adecuado y recolecta resultados
                if return_features:
                    # Devuelve tanto características internas como salidas
                    features, outputs = self.model(inputs, mode='both')
                    all_features.append(features.cpu())
                    all_outputs.append(outputs.cpu())
                    
                else:
                    # Solo devuelve la salida final del modelo
                    outputs = self.model(inputs, mode='features')
                    all_outputs.append(outputs.cpu())
                    
        # Concatena los resultados de todos los batches
        result = []
        if has_targets:
            result.append(torch.cat(all_targets))
        if return_features:
            result.append(torch.cat(all_features))
        result.append(torch.cat(all_outputs))

        # Devuelve una tupla con los resultados o solo un valor si hay uno
        return result[0] if len(result) == 1 else tuple(result)
    

    def evaluate(self, dataloader, metric_fn):
        
        #
        all_targets, all_predicted = self.inference(dataloader)
        
        #
        metric_value = metric_fn(all_predicted, all_targets)
        return metric_value
    
    
    def train_head(self, train_loader, valid_loader, loss_fn, optimizer, num_epochs=1):
        
        # Congela los parámetros del extractor de características
        for param in self.model.head_parameters():
            param.requires_grad = False
            
        # Configura el optimizador para el entrenamiento de la nueva cabecera (el módulo classifier)
        optimizer = torch.optim.AdamW(self.model.get_head_parameters(), lr=base_lr, weight_decay=wd)
        
        for epoch in range(num_epochs):

            # Entrena el modelo con el conjunto de entrenamiento
            head_train_loss = self.train_epoch(train_loader, loss_fn, optimizer)

            # Evalua el modelo con el conjunto de validación
            head_valid_loss = self.evaluate(valid_loader, loss_fn)

            # Imprime los valores de pérdida obtenidos en entrenamiento y validación 
            print(f"Epoch {epoch+1:>2} | " +
                  f"Train Loss: {head_train_loss:>7.3f} | " + 
                  f"Validation Loss: {head_valid_loss:>7.3f}")

    
    def train_net(self, train_loader, valid_loader, loss_fn, optimizer, scheduler):
        
        #-----------------------------------------------------------------------------------------------------
        # DESCONGELA PARÁMETROS DEL MODELO 
        
        # Descongela todos los parámetros del modelo
        for param in self.model.parameters():
            param.requires_grad = True
        
        #-----------------------------------------------------------------------------------------------------
        # ASIGNA LEARNING RATE DISCRIMINATIVO
        
        # Crea una lista para almacenar los nombres de las capas del modelo
        layer_names = []
        for (name, param) in self.model.named_parameters():
            layer_names.append(name)

        # Establece las reglas para el learning rate discriminativo   
        lr_div = 100            # Factor de reducción entre el learning rate más alto y el más pequeño
        max_lr = base_lr/2      # Learning rate más alto (capa más superficial)
        min_lr = max_lr/lr_div  # Learning rate más bajo (capa más profunda) 
        
        # Obtenemos los grupos de capas del modelo (de más superficiales a más profundas)
        layer_groups = self.model.get_layer_groups()
        n_layers = len(layer_groups)

        # Genera una lista de tasas de aprendizaje para cada capa, aumentando de forma exponencial desde 
        # min_lr hasta max_lr
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
            
        #-----------------------------------------------------------------------------------------------------
        # INICIALIZA LOS PARÁMETROS PARA EL EARLY STOPPING
        
        # Número máximo de épocas a entrenar (si no se activa el early stopping)
        MAX_EPOCHS = 30  

        # Número mínimo de épocas a entrenar
        MIN_EPOCHS = 30

        # Número de épocas sin mejora antes de detener el entrenamiento
        PATIENCE = 10

        # Inicializa la mejor pérdida de validación como la obtenida en el entrenamiento de la cabecera
        best_valid_loss = float('inf') 

        # Contador de épocas sin mejora
        epochs_no_improve = 0 
            
        #-----------------------------------------------------------------------------------------------------
        # ENTRENAMIENTO DE LA RED COMPLETA
        
        # Listas para almacenar las pérdidas de entrenamiento y validación
        train_losses = []
        valid_losses = []

        # Bucle de entrenamiento por épocas
        for epoch in range(MAX_EPOCHS):
            
            # Entrena el modelo con el conjunto de entrenamiento
            train_loss = self.train_epoch(train_loader, loss_fn, optimizer, scheduler)
            train_losses.append(train_loss)
            
            # Evalua el modelo con el conjunto de validación
            valid_loss = self.evaluate(valid_loader, loss_fn)
            valid_losses.append(valid_loss)
            
            # Imprime los valores de pérdida obtenidos en entrenamiento y validación  
            print(f"Epoch {epoch+1:>2} | " +
                  f"Train Loss: {train_loss:>7.3f} | " +
                  f"Validation Loss: {valid_loss:>7.3f}")
            
            # Comprueba si la pérdida en validación ha mejorado
            if valid_loss < best_valid_loss:
                
                # Actualiza la mejor pérdida en validación obtenida hasta ahora
                best_valid_loss = valid_loss
                
                # Reinicia el contador de épocas sin mejora si la pérdida ha mejorado
                epochs_no_improve = 0
                
                # 
                self.save_model(self.default_model_path)
                
            else:
                # Incrementa el contador si no hay mejora en la pérdida de validación
                epochs_no_improve += 1

            # Si no hay mejora durante un número determinado de épocas (patience) y ya ha pasado el número mínimo de 
            # épocas, detiene el entrenamiento
            if epochs_no_improve >= PATIENCE and (epoch+1) > MIN_EPOCHS: 
                print(f"Early stopping at epoch {epoch+1}")
                break
            
        # Carga los pesos del modelo que obtuvo la mejor validación
        self.load_model(self.default_model_path)



class AE_ModelRunner_QR(AE_ModelRunner):

    PRED_MODEL_TYPE = 'QR'

    def __init__(self, model, alpha=0.9, model_path=None, device=None):
        
        super.__init__(model, model_path=model, device=device)
        
        self.alpha = alpha


    def save_model(self, model_path=None):
        
        model_path = self.default_model_path if model_path is None else model_path
        
        # Crea el checkpoint con el tipo de modelo y sus pesos
        checkpoint = {
            'pred_model_type': self.PRED_MODEL_TYPE,
            'model_state_dict': self.model.state_dict()
        }
        
        # Guarda el checkpoint 
        torch.save(checkpoint, model_path)


    def load_model(self, model_path=None):
        """Carga un checkpoint del modelo, incluyendo parámetros de calibración si se desea."""
        
        model_path = self.default_model_path if model_path is None else model_path
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self._verify_checkpoint_type(checkpoint)

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.alpha = checkpoint['alpha']

        except Exception as e:
            raise RuntimeError(f"Error al cargar el modelo QR desde {model_path}: {e}") from e



class AE_ModelRunner_ICP(AE_ModelRunner):

    PRED_MODEL_TYPE = 'ICP'

    def __init__(self, model, alpha=0.9, model_path=None, device=None):
        
        super.__init__(model, model_path=model, device=device)
        
        self.alpha = alpha


    def save_model(self, include_calib_parameters=True, model_path=None):
        
        model_path = self.default_model_path if model_path is None else model_path
        
        # Crea el checkpoint con el tipo de modelo y sus pesos
        checkpoint = {
            'pred_model_type': self.PRED_MODEL_TYPE,
            'model_state_dict': self.model.state_dict(),
            'alpha': self.alpha
        }
        
        if include_calib_parameters:
            checkpoint['q_hat'] = self.q_hat
        
        # Guarda el checkpoint 
        torch.save(checkpoint, model_path)


    def load_model(self, include_calib_parameters=True, model_path=None):
        """Carga un checkpoint del modelo, incluyendo parámetros de calibración si se desea."""
        
        model_path = self.default_model_path if model_path is None else model_path
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self._verify_checkpoint_type(checkpoint)

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.alpha = checkpoint['alpha']

            if include_calib_parameters:
                self.q_hat = checkpoint['q_hat']

        except Exception as e:
            raise RuntimeError(f"Error al cargar el modelo ICP desde {model_path}: {e}") from e


    def calibrate(self, calib_loader):
        
        # Obtener predicciones y valores verdaderos del conjunto de calibración
        calib_true_values, calib_pred_values = self.model.inference(calib_loader)
        
        # Calcula el nivel de cuantificación ajustado basado en el tamaño del conjunto de calibración y alpha
        n = len(calib_true_values)
        q_level = np.ceil((1.0 - self.alpha) * (n + 1.0)) / n
        
        # Calcula las puntuaciones de calibración como valores absolutos de los errores
        calib_scores = np.abs(calib_true_values-calib_pred_values)
        
        # Calcula el cuantil q_hat usado para ajustar el intervalo predictivo
        self.q_hat = np.quantile(calib_scores, q_level, method='higher')



class AE_ModelRunner_CQR(AE_ModelRunner):

    PRED_MODEL_TYPE = 'CQR'

    def __init__(self, model, alpha=0.9, model_path=None, device=None):
        
        super.__init__(model, model_path=model, device=device)
        
        self.alpha = alpha


    def save_model(self, include_calib_parameters=True, model_path=None):
        
        model_path = self.default_model_path if model_path is None else model_path
        
        # Crea el checkpoint con el tipo de modelo y sus pesos
        checkpoint = {
            'pred_model_type': self.PRED_MODEL_TYPE,
            'model_state_dict': self.model.state_dict(),
            'alpha': self.alpha
        }
        
        if include_calib_parameters:
            checkpoint['q_hat_lower'] = self.q_hat_lower
            checkpoint['q_hat_upper'] = self.q_hat_upper
        
        # Guarda el checkpoint 
        torch.save(checkpoint, model_path)


    def load_model(self, include_calib_parameters=True, model_path=None):
        """Carga un checkpoint del modelo, incluyendo parámetros de calibración si se desea."""
        
        model_path = self.default_model_path if model_path is None else model_path
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self._verify_checkpoint_type(checkpoint)

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.alpha = checkpoint['alpha']

            if include_calib_parameters:
                self.q_hat_lower = checkpoint['q_hat_lower']
                self.q_hat_upper = checkpoint['q_hat_upper']

        except Exception as e:
            raise RuntimeError(f"Error al cargar el modelo CQR desde {model_path}: {e}") from e


    def calibrate(self, calib_loader):
        
        # Obtener predicciones y valores verdaderos del conjunto de calibración
        calib_true_values, calib_pred_values = self.model.inference(calib_loader)
        
        # Calcula el nivel de cuantificación ajustado basado en el tamaño del conjunto de calibración y alpha
        n = len(calib_true_values)
        q_level = np.ceil((1.0 - (self.alpha / 2.0)) * (n + 1.0)) / n

        # Calcula las puntuaciones para el límite inferior (diferencia entre predicción inferior y valor real)
        # y para el límite superior (diferencia entre valor real y predicción superior)
        calib_scores_lower_bound = calib_pred_values[:, 1] - calib_true_values
        calib_scores_upper_bound = calib_true_values - calib_pred_values[:,2]
        
        # Calcula los cuantiles qhat para ambos límites del intervalo predictivo
        q_hat_lower = np.quantile(calib_scores_lower_bound, q_level, method='higher')
        q_hat_upper = np.quantile(calib_scores_upper_bound, q_level, method='higher')



class AE_ModelRunner_CRF(AE_ModelRunner):

    PRED_MODEL_TYPE = 'CRF'

    def __init__(self, model, alpha=0.9, model_path=None, device=None):
        
        super.__init__(model, model_path=model, device=device)
        
        self.alpha = alpha


    def save_model(self, include_calib_parameters=True, model_path=None):
        
        model_path = self.default_model_path if model_path is None else model_path
        
        # Crea el checkpoint con el tipo de modelo y sus pesos
        checkpoint = {
            'pred_model_type': self.PRED_MODEL_TYPE,
            'model_state_dict': self.model.state_dict(),
            'alpha': self.alpha
        }
        
        #-------
        
        # Guarda el checkpoint 
        torch.save(checkpoint, model_path)


    def load_model(self, include_calib_parameters=True, model_path=None):
        """Carga un checkpoint del modelo, incluyendo parámetros de calibración si se desea."""
        
        model_path = self.default_model_path if model_path is None else model_path
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self._verify_checkpoint_type(checkpoint)

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.alpha = checkpoint['alpha']

            #-------

        except Exception as e:
            raise RuntimeError(f"Error al cargar el modelo CRF desde {model_path}: {e}") from e


    def calibrate(self, calib_loader):
        
        #-------
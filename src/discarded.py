def enable_dropout(model, p=None):
    """
    Activa todos los dropout layers durante la inferencia.
    Si se pasa p, modifica la probabilidad de dropout.
    """
    for m in model.modules():
        if isinstance(m, nn.Dropout):  # Más robusto que comparar el nombre de clase
            m.train()
            if p is not None:
                m.p = p  # Cambia la probabilidad de dropout


def mc_dropout_inference(model, dataloader, patience=5, min_delta=0.01, max_mc=100, p=0.5, device='cuda'):
    """
    Realiza inferencia con MC Dropout dinámico, deteniéndose cuando la varianza converge.

    Parámetros:
    - model: el modelo de PyTorch
    - dataloader: el dataloader con los datos de prueba
    - patience: número de iteraciones consecutivas en las que la varianza debe ser estable
    - min_delta: umbral mínimo de cambio en la varianza para considerarla estable
    - max_mc: número máximo de muestras MC por ejemplo
    - p: probabilidad de dropout (si queremos modificarla en inferencia)
    - device: 'cuda' o 'cpu'.
    """
    
    model.eval()
    enable_dropout(model, p)
    
    all_samples = []
    all_targets = []
    
    # No calculamos gradientes (más rápido y consume menos memoria)
    with torch.no_grad():
        for inputs, targets in dataloader:
            
            # Obtiene las imágenes y sus valores objetivo
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 
            batch_size = inputs.shape[0]
            
            # Almacena las predicciones MC aquí
            preds = []
            
            # Contadores de paciencia por cada instancia del batch
            current_patience_count = torch.zeros(batch_size, dtype=torch.int, device=device)
            
            # Almacena la varianza de la iteración anterior para comparar
            prev_variance = None
            
            #
            for mc_iter in range(max_mc):
                
                # Hace forward pass con dropout activado
                outputs = model(inputs) # [total_batch_size, output_dim]
                
                # Añade una dimensión para acumular fácilmente
                preds.append(outputs.unsqueeze(0))
                
                # Convertimos la lista de predicciones a tensor de shape [mc_iter+1, batch_size, output_dim]
                preds_tensor = torch.cat(preds, dim=0)
                
                #  Calculamos la varianza de las predicciones actuales a lo largo de la dimensión MC
                variance = preds_tensor.std(dim=0) # Resultado: [batch_size, output_dim]
                
                if mc_iter > 0:
                    # Si ya tenemos al menos dos muestras, calculamos la diferencia de varianza
                    var_diff = torch.abs(variance - prev_variance)  # [batch_size, output_dim]
                    
                    # Comprobamos, para cada muestra del batch, si todas sus salidas están dentro de min_delta
                    stable = (var_diff <= min_delta).all(dim=1)  # Resultado: [batch_size], True si estable
                    
                    # Si es estable, aumenta la paciencia; si no, se resetea a 0
                    current_patience_count += stable.int()
                    current_patience_count *= stable.int()      
                
                # Guarda la varianza actual para comparar en la próxima iteración
                prev_variance = variance
                
                # Si todos los ejemplos del batch han superado el patience, para el bucle
                if (current_patience_count > patience).all():
                    break
            
            # Cuando termina el bucle (por paciencia o por max_mc), calcula la media MC
            preds_tensor = preds_tensor.mean(dim=0)
            
            # Guarda predicciones y targets para este batch
            all_predicted.append(preds_tensor.cpu())
            all_targets.append(targets.cpu())
            
    # Une todos los batches en un único tensor
    all_predicted = torch.cat(all_predicted, dim=0) # [total_batch_size, output_dim]
    all_targets = torch.cat(all_targets)
        
    return all_predicted, all_targets
#-------------------------------------------------------------------------------------------------------------
# BIBLIOTECAS ------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------

import torch

#-------------------------------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------------------------------
# METRICS FOR CONFORMAL REGRESSION ---------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------

def empirical_coverage(
    pred_lower_bound : torch.Tensor,
    pred_upper_bound : torch.Tensor,
    true_values : torch.Tensor
) -> int:
    
    inside_interval = (true_values >= pred_lower_bound) & (true_values <= pred_upper_bound)
    empirical_coverage = inside_interval.float().mean().item()
    return empirical_coverage

# ------------------------------------------------------------------------------------------------------------

def mean_interval_size(
    pred_lower_bound : torch.Tensor,
    pred_upper_bound : torch.Tensor,
) -> int:
    
    # Calcula el ancho promedio de los intervalos y lo devuelve
    mean_interval_size = (pred_upper_bound - pred_lower_bound).mean().item()
    return mean_interval_size

# ------------------------------------------------------------------------------------------------------------

def quantile_interval_size(
    pred_lower_bound : torch.Tensor,
    pred_upper_bound : torch.Tensor,
    quantile: float
) -> torch.Tensor:

    interval_sizes = (pred_upper_bound - pred_lower_bound)
    quantile_value = torch.quantile(interval_sizes, quantile)
    return quantile_value

# ------------------------------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------------------------------
# METRICS FOR CONFORMAL CLASSIFICATION -----------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------

def empirical_coverage_classification(
    pred_sets : torch.Tensor,
    true_labels : torch.Tensor  
) -> int:
    """
    Calcula la cobertura empírica para clasificación usando predicciones conformales.

    Parámetros:
    - true_labels (torch.Tensor): Tensor de forma (n,) con etiquetas verdaderas (enteros de clase).
    - pred_sets (torch.Tensor): Tensor booleano o binario de forma (n, num_classes), donde 
      pred_sets[i, c] = True/1 indica que la clase 'c' está incluida en el conjunto de predicciones para el 
      ejemplo i.

    Devuelve:
    - torch.Tensor: Cobertura empírica como un escalar (proporción de ejemplos donde la etiqueta verdadera
      está incluida en el conjunto de predicción correspondiente).
    """
    
    # Obtener índice de fila y clase verdadera
    row_indices = torch.arange(true_labels.shape[0])
    # Verifica si la etiqueta verdadera está presente en el conjunto predicho
    covered = pred_sets[row_indices, true_labels]
    # Promedio de coberturas (float tensor)
    return covered.float().mean().item()

# ------------------------------------------------------------------------------------------------------------

def empirical_coverage_by_class(
    pred_sets: torch.Tensor,
    true_labels: torch.Tensor,
    num_classes: int
) -> torch.Tensor:
    """
    Calcula la cobertura media específica por clase.

    Parámetros:
    - true_labels (torch.Tensor): Tensor de forma (n,) con etiquetas verdaderas.
    - pred_sets (torch.Tensor): Tensor booleano o binario de forma (n, num_classes),
      donde cada fila indica qué clases están en el conjunto predicho para ese ejemplo.
    - num_classes (int): Número total de clases.
    
    Devuelve:
    - torch.Tensor: Tensor 1D de tamaño (num_classes,) con cobertura media por clase.
      Si una clase no está presente en true_labels, su cobertura será 0.0.
    """
    # Obtener cantidad total por clase
    total_per_class = torch.bincount(true_labels, minlength=num_classes)

    # Crear tensor booleano (n,): True si la clase verdadera está en el conjunto predicho
    row_indices = torch.arange(true_labels.shape[0])
    covered_mask = pred_sets[row_indices, true_labels]

    # Sumar cuántos ejemplos por clase fueron correctamente cubiertos
    covered_per_class = torch.bincount(
        true_labels[covered_mask],
        minlength=num_classes
    )

    # Evitar división por cero
    coverage = torch.zeros(num_classes, dtype=torch.float)
    mask_nonzero = total_per_class > 0
    coverage[mask_nonzero] = covered_per_class[mask_nonzero].float() / total_per_class[mask_nonzero]

    return coverage

# ------------------------------------------------------------------------------------------------------------

def mean_set_size(
    pred_sets: torch.Tensor
) -> float:
    """
    Calcula el tamaño promedio de los conjuntos de predicción.

    Parámetros:
    - pred_sets (torch.Tensor): Tensor booleano o binario de forma (n, num_classes),
      donde cada fila representa el conjunto de predicción para un ejemplo.

    Devuelve:
    - torch.Tensor: Un escalar con el tamaño promedio de los conjuntos de predicción.
    """
    # Sumar valores por fila => tamaño del conjunto para cada ejemplo
    set_sizes = pred_sets.sum(dim=1)
    # Calcular la media de los tamaños
    return set_sizes.float().mean().item()

# ------------------------------------------------------------------------------------------------------------

# def mean_set_score(
#     pred_sets: torch.Tensor,
#     true_labels: torch.Tensor,
#     alpha: float
# ) -> float:
#     """
#     Calcula ...
#     """
#     # Obtener índice de fila y clase verdadera
#     row_indices = torch.arange(true_labels.shape[0])
#     # Verifica si la etiqueta verdadera está presente en el conjunto predicho
#     covered = pred_sets[row_indices, true_labels]
    
#     MSS = pred_sets.sum(dim=1) + 1/alpha * (~covered)
    
#     # Calcular la media de los tamaños
#     return MSS.float().mean().item()

# ------------------------------------------------------------------------------------------------------------

def mean_set_size_by_class(
    pred_sets: torch.Tensor
) -> torch.Tensor:
    """
    Calcula el tamaño promedio del conjunto de predicción para cada clase.

    Parámetros:
    - pred_sets (torch.Tensor): Tensor booleano o binario de forma (n, num_classes),
      donde pred_sets[i, c] indica si la clase c está en el conjunto predicho para el ejemplo i.
    - num_classes (int): Número total de clases.

    Devuelve:
    - torch.Tensor: Tensor de tamaño (num_classes,) con la frecuencia promedio de aparición de cada clase.
    """
    # Sumar ocurrencias por clase (columna)
    class_counts = pred_sets.sum(dim=0)
    # Dividir por número total de ejemplos
    return class_counts.float() / pred_sets.shape[0]

# ------------------------------------------------------------------------------------------------------------
# Para clasificación binaria

def indeterminancy_rate(
    pred_sets: torch.Tensor
) -> float:
    
    # Suma valores por fila => tamaño del conjunto para cada ejemplo
    set_sizes = pred_sets.sum(dim=1)
    # Calcula el porcentaje de instancias con más de una clase predicha
    indeterminate = (set_sizes > 1).sum().item()
    total = pred_sets.size(0)
    return indeterminate / total

# ------------------------------------------------------------------------------------------------------------
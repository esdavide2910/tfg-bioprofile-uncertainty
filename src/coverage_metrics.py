import torch
import matplotlib.pyplot as plt
import numpy as np


def empirical_coverage(
    pred_lower_bound : torch.Tensor,
    pred_upper_bound : torch.Tensor,
    true_values : torch.Tensor
) -> torch.Tensor:
    
    inside_interval = (true_values >= pred_lower_bound) & (true_values <= pred_upper_bound)
    
    empirical_coverage = inside_interval.float().mean().item()
    
    return empirical_coverage


def mean_interval_size(
    pred_lower_bound : torch.Tensor,
    pred_upper_bound : torch.Tensor,
) -> torch.Tensor:
    
    # Calcula el ancho promedio de los intervalos y lo devuelve
    mean_interval_size = (pred_upper_bound - pred_lower_bound).float().mean().item()
    return mean_interval_size


def quantile_interval_size(
    pred_lower_bound : torch.Tensor,
    pred_upper_bound : torch.Tensor,
    quantile: float
) -> torch.Tensor:

    interval_sizes = (pred_upper_bound - pred_lower_bound).float()
    quantile_value = torch.quantile(interval_sizes, quantile)
    return quantile_value


def plot_confidence_predictions(
    pred_point_values: torch.Tensor,
    pred_lower_bound: torch.Tensor,
    pred_upper_bound: torch.Tensor,
    true_values: torch.Tensor,
):
    
    # Convertimos los tensores a NumPy
    y_true = true_values.detach().numpy()
    y_pred = pred_point_values.detach().numpy()
    y_lower = pred_lower_bound.detach().numpy()
    y_upper = pred_upper_bound.detach().numpy()
    
    # Ordenamos por los valores verdaderos
    order = np.argsort(y_true)
    y_true = y_true[order]
    y_pred = y_pred[order]
    y_lower = y_lower[order]
    y_upper = y_upper[order]
    
    x = np.arange(len(y_true))
    
    # Graficamos
    plt.figure(figsize=(12, 6))
    plt.plot(x, y_true, label='Valor verdadero', color='black', linewidth=2)
    plt.plot(x, y_pred, label='Predicción puntual', color='blue', linestyle='--')
    plt.fill_between(x, y_lower, y_upper, color='lightblue', alpha=0.5, label='Intervalo de confianza')

    plt.xlabel('Instancia (ordenada por valor verdadero)')
    plt.ylabel('Valor')
    plt.title('Predicción con intervalos de confianza')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



import torch


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

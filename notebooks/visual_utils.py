import torch
import numpy as np
import polars as pl 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Iterable, Tuple, Optional, Union

def plot_coverage_vs_interval_width(
    mean_pred_interval_widths: Iterable[float], 
    empirical_coverages: Iterable[float], 
    model_types: Iterable[str],
    colors: Iterable[str],
    confidence_level: float = None,
    figsize: Tuple[float, float] = (8, 6)
) -> None:
    """
    Dibuja un scatterplot de Empirical Coverage vs. Mean Prediction Interval Width,
    coloreando los puntos según el tipo de modelo.
    
    Args:
        mean_pred_interval_widths (Iterable[float]): valores para el eje X
        empirical_coverages (Iterable[float]): valores para el eje Y
        model_types (Iterable[str]): etiquetas de modelo para cada punto
        colors (Iterable[str]): lista de colores para cada punto
        confidence_level (float): Línea horizontal de referencia para la cobertura deseada. Por defecto: None
        figsize (Tuple[float, float]): Tamaño de la figura en pulgadas, como (ancho, alto). Por defecto: (8, 6)
    """

    # Convertimos a listas para poder indexar y hacer validaciones
    mean_pred_interval_widths = list(mean_pred_interval_widths)
    empirical_coverages = list(empirical_coverages)
    model_types = list(model_types)
    colors = list(colors)

    if not (len(mean_pred_interval_widths) == len(empirical_coverages) == len(model_types) == len(colors)):
        raise ValueError("All input iterables must have the same length.")
    
    plt.figure(figsize=figsize)
    plt.grid(True, zorder=0)

    # Graficar todos los puntos como círculos
    plt.scatter(
        mean_pred_interval_widths,
        empirical_coverages,
        c=colors,
        marker='o',
        s=100,
        edgecolor='white',
        linewidth=0.5,
        zorder=3
    )
    
    # Línea horizontal en y=confidence_level
    if confidence_level is not None:
        plt.axhline(y=confidence_level, color='black', linestyle='--', linewidth=1.0, zorder=2)

    # Etiquetas y título
    plt.xlabel("Mean Prediction Interval Width")
    plt.ylabel("Empirical Coverage")
    plt.title("Empirical Coverage vs. Mean Prediction Interval Width")

    # Legend
    seen = set()
    legend_elements = []
    for model, color in zip(model_types, colors):
        if model not in seen:
            legend_elements.append(
                plt.Line2D([0], [0], marker='o', color='w', label=model,
                           markerfacecolor=color, markersize=10, markeredgecolor='white')
            )
            seen.add(model)

    plt.legend(handles=legend_elements, title="Model Type")
    plt.tight_layout()
    plt.show()


def plot_coverage_vs_set_size(
    mean_set_sizes: Iterable[float], 
    empirical_coverages: Iterable[float], 
    model_types: Iterable[str],
    colors: Iterable[str],
    confidence_level: float = None,
    figsize: Tuple[float, float] = (8, 6)
) -> None:
    """
    """

    # Convertimos a listas para poder indexar y hacer validaciones
    mean_pred_set_sizes = list(mean_set_sizes)
    empirical_coverages = list(empirical_coverages)
    model_types = list(model_types)
    colors = list(colors)

    if not (len(mean_pred_set_sizes) == len(empirical_coverages) == len(model_types) == len(colors)):
        raise ValueError("All input iterables must have the same length.")
    
    plt.figure(figsize=figsize)
    plt.grid(True, zorder=0)

    # Graficar todos los puntos como círculos
    plt.scatter(
        mean_pred_set_sizes,
        empirical_coverages,
        c=colors,
        marker='o',
        s=100,
        edgecolor='white',
        linewidth=0.5,
        zorder=3
    )
    
    # Línea horizontal en y=confidence_level
    if confidence_level is not None:
        plt.axhline(y=confidence_level, color='black', linestyle='--', linewidth=1.0, zorder=2)

    # Etiquetas y título
    plt.xlabel("Mean Prediction Set Sizes")
    plt.ylabel("Empirical Coverage")
    plt.title("Empirical Coverage vs. Mean Prediction Set Sizes")

    # Legend
    seen = set()
    legend_elements = []
    for model, color in zip(model_types, colors):
        if model not in seen:
            legend_elements.append(
                plt.Line2D([0], [0], marker='o', color='w', label=model,
                           markerfacecolor=color, markersize=10, markeredgecolor='white')
            )
            seen.add(model)

    plt.legend(handles=legend_elements, title="Model Type")
    plt.tight_layout()
    plt.show()


def sort_by_column(
    df: pl.DataFrame,
    column: str,
    order: Optional[Iterable[Union[str, int]]] = None
) -> pl.DataFrame:
    """
    Ordena un DataFrame de Polars según un orden personalizado en una columna,
    o por orden natural si no se especifica un orden.

    Parámetros:
    - df: DataFrame de Polars.
    - column: nombre de la columna a ordenar.
    - order: iterable con el orden deseado de los valores, o None para orden natural.

    Retorna:
    - DataFrame ordenado.
    """
    if order is not None:
        # Convertir a lista para usar .index()
        order_list = list(order)

        # Validación: asegurar que todos los valores están en el orden
        unique_vals = set(df[column].to_list())
        missing = unique_vals - set(order_list)
        if missing:
            raise ValueError(f"Valores no presentes en 'order': {missing}")

        return (
            df.with_columns(
                pl.col(column).map_elements(lambda x: order_list.index(x), return_dtype=pl.Int32).alias("_custom_sort_index")
            )
            .sort("_custom_sort_index")
            .drop("_custom_sort_index")
        )
    else:
        # Orden natural: string alfabético o numérico ascendente
        return df.sort(column)


def plot_confidence_predictions(
    pred_point_values : torch.Tensor,
    pred_lower_bound : torch.Tensor,
    pred_upper_bound : torch.Tensor,
    true_values : torch.Tensor
) -> None:
    
    # Convierte los tensores a NumPy
    y_true = true_values.numpy()
    y_pred = pred_point_values.numpy()
    y_lower = pred_lower_bound.numpy()
    y_upper = pred_upper_bound.numpy()
    
    # Ordena por los valores predichos
    order = np.argsort(y_pred)
    y_true = y_true[order]
    y_pred = y_pred[order]
    y_lower = y_lower[order]
    y_upper = y_upper[order]
    
    x = np.arange(len(y_true))
    
    # Grafica
    plt.figure(figsize=(12, 6))
    plt.plot(x, y_true, label='Valor verdadero', 
             color='black', linewidth=2, marker='s', markersize=4)
    plt.plot(x, y_pred, label='Predicción puntual', 
             color='steelblue', linestyle='--', marker='o', markersize=4)
    plt.fill_between(x, y_lower, y_upper, 
                     color='lightcoral', alpha=0.5, label='Intervalo de confianza')

    plt.xlabel('Instancia (ordenada por valor verdadero)')
    plt.ylabel('Valor')
    plt.title('Predicción con intervalos de confianza')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
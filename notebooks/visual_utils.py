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
    figsize: Tuple[float, float] = (8, 6),
    dpi: int = 100 
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
    
    plt.figure(figsize=figsize, dpi=dpi)
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
    plt.xlabel("Mean Prediction Interval Width", fontsize=12)
    plt.ylabel("Empirical Coverage", fontsize=12)
    plt.title("Empirical Coverage vs. Mean Prediction Interval Width", pad=15)

    # Legend
    seen = set()
    legend_elements = []
    for model, color in zip(model_types, colors):
        if model not in seen:
            legend_elements.append(
                plt.Line2D([0], [0], marker='o', color='w', label=model,
                           markerfacecolor=color, markersize=8, markeredgecolor='white')
            )
            seen.add(model)
    
    # Obtener los límites actuales del eje Y
    ymin, ymax = plt.gca().get_ylim()

    # Ajustar los ticks cada 0.01 dentro del rango visible
    plt.yticks(np.arange(np.floor(ymin*100)/100, np.ceil(ymax*100)/100 + 0.01, 0.01))

    plt.legend(handles=legend_elements, title="Model Type")
    plt.tight_layout()
    plt.show()


def plot_coverage_vs_set_size(
    mean_set_sizes: Iterable[float], 
    empirical_coverages: Iterable[float], 
    model_types: Iterable[str],
    colors: Iterable[str],
    confidence_level: float = None,
    figsize: Tuple[float, float] = (8, 6),
    dpi: int = 100 
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
    
    plt.figure(figsize=figsize, dpi=dpi)
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
    plt.xlabel("Mean Prediction Set Sizes", fontsize=12)
    plt.ylabel("Empirical Coverage", fontsize=12)
    plt.title("Empirical Coverage vs. Mean Prediction Set Sizes", pad=15)

    # Legend
    seen = set()
    legend_elements = []
    for model, color in zip(model_types, colors):
        if model not in seen:
            legend_elements.append(
                plt.Line2D([0], [0], marker='o', color='w', label=model,
                           markerfacecolor=color, markersize=8, markeredgecolor='white')
            )
            seen.add(model)
    
    # Obtener los límites actuales del eje Y
    ymin, ymax = plt.gca().get_ylim()

    # Ajustar los ticks cada 0.01 dentro del rango visible
    plt.yticks(np.arange(np.floor(ymin*100)/100, np.ceil(ymax*100)/100 + 0.01, 0.01))

    plt.legend(handles=legend_elements, title="Model Type")
    plt.tight_layout()
    plt.show()


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



def plot_interval_width_histogram(
    df: pd.DataFrame, 
    pred_model_type: str, 
    confidence: float,
    bins: int = 30
) -> plt.Figure:
    """
    Genera un histograma apilado del ancho de intervalos, diferenciando entre coberturas correctas y fallidas.
    """
    # Filtrar datos y asegurarse de que existe la columna is_covered
    filtered_data = df[
        (df['pred_model_type'] == pred_model_type) & 
        (df['confidence'] == confidence)
    ].copy()
    
    if len(filtered_data) == 0:
        raise ValueError(f"No hay datos para {pred_model_type} con confianza {confidence}")
    
    if 'is_covered' not in filtered_data.columns:
        raise ValueError("El DataFrame debe contener la columna 'is_covered'")
    
    # Convertir booleanos a strings para mejor visualización
    filtered_data['coverage_status'] = filtered_data['is_covered'].map({
        True: 'Cubre valor real',
        False: 'No cubre valor real'
    })
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Histograma apilado con hue
    sns.histplot(
        data=filtered_data,
        x='pred_interval_width',
        hue='coverage_status',
        bins=bins,
        multiple='stack',
        palette=['#4CAF50', '#F44336'],  # Verde y rojo
        edgecolor='white',
        linewidth=0.5,
        ax=ax,
        kde=True,
        legend=True
    )
    
    # Calcular estadísticos
    coverage_rate = filtered_data['is_covered'].mean()
    
    # Personalización
    ax.set_title(
        f"Distribución del ancho de intervalo\n"
        f"Modelo: {pred_model_type} | Confianza: {confidence}\n"
        f"Cobertura empírica: {coverage_rate:.1%}",
        pad=20
    )
    ax.set_xlabel("Ancho del intervalo de predicción")
    ax.set_ylabel("Frecuencia")
    
    plt.tight_layout()
    return fig
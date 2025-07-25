import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Iterable, Tuple, Optional, Union

#-------------------------------------------------------------------------------------------

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
    plt.xlabel("Amplitud Media de Intervalo de Predicción", fontsize=12)
    plt.ylabel("Cobertura empírica", fontsize=12)
    # plt.title("Cobertura Empírica vs. Amplitud Media de Intervalo de Predicción", pad=15)

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

    plt.legend(handles=legend_elements, title="Método")
    plt.tight_layout()
    plt.show()

#-------------------------------------------------------------------------------------------

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
    plt.xlabel("Tamaño Medio de Conjunto de Predicción", fontsize=12)
    plt.ylabel("Cobertura empírica", fontsize=12)
    # plt.title("Cobertura empírica vs. Tamaño Medio de Conjunto de Predicción", pad=15)

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

    plt.legend(handles=legend_elements, title="Método")
    plt.tight_layout()
    plt.show()

#-------------------------------------------------------------------------------------------

def plot_interval_width_histogram(
    df: pd.DataFrame, 
    pred_model_type: str, 
    confidence: float,
    bins: int = 30,
    dpi=120
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
    fig, ax = plt.subplots(figsize=(10, 6.2), dpi=dpi)
    
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
    ax.set_xlabel("Amplitud del intervalo de predicción", fontsize=14)
    ax.set_ylabel("Frecuencia", fontsize=14)
    
    # Cambiar tamaño de xticks e yticks
    ax.tick_params(axis='both', labelsize=13)
    
    plt.tight_layout()
    return fig

#-------------------------------------------------------------------------------------------

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

#-------------------------------------------------------------------------------------------

def plot_image(
        image, label=None, ax=None, size=3.5, font_size=10
    ):
    """
    Dibuja una imagen en un eje dado con su etiqueta correspondiente.
    Si no se proporciona un eje, se crea una nueva figura y eje con el tamaño especificado,
    asegurando que la dimensión más larga tenga la longitud `size`.
    Si la imagen tiene un solo canal, se muestra en escala de grises.
    """
    # Convertir imágenes PIL a NumPy
    if isinstance(image, Image.Image):
        image = np.array(image)  # Convertir PIL a NumPy

    # Convertir tensores de PyTorch a NumPy
    if isinstance(image, torch.Tensor):  # Si es un tensor de PyTorch
        image = image.detach().cpu().numpy()  # Convertir a NumPy
        if image.ndim == 3 and image.shape[0] in [1, 3]:  # Si está en formato (C, H, W)
            image = np.transpose(image, (1, 2, 0))  # Pasar a formato (H, W, C)
        elif image.ndim == 2:  # Imagen en escala de grises
            pass  # No necesita transformación

    if ax is None:
        # Obtener dimensiones de la imagen
        height, width = image.shape[:2]
        aspect_ratio = width / height

        # Ajustar el tamaño de la figura manteniendo la proporción dentro del cuadrado size x size
        if aspect_ratio > 1:  # Imagen apaisada
            fig_size = (size, size / aspect_ratio)
        else:  # Imagen vertical o cuadrada
            fig_size = (size * aspect_ratio, size)

        fig, ax = plt.subplots(figsize=fig_size)
        own_ax = True
    else:
        own_ax = False

    # Determinar si la imagen es en escala de grises
    cmap = "gray" if image.ndim == 2 or (image.ndim == 3 and image.shape[-1] == 1) else None

    ax.imshow(image.squeeze(), cmap=cmap)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_axis_off()

    # Si la imagen tiene una etiqueta, se añade como título
    if label is not None:
        ax.set_title(label, fontsize=font_size)

    # Si se creó un eje nuevo, mostrar la figura
    if own_ax:
        plt.show()

#-------------------------------------------------------------------------------------------

def plot_images(
        images, labels=None,
        images_per_row=4, size=3.5, font_size=25,
        vertical_spacing=0.1, horizontal_spacing=0.1,
        fig_aspect=(1,1)
    ):
    """
    Dibuja un conjunto de imágenes con etiquetas en una cuadrícula.

    Args:
      - images (list or np.array): Lista o array de imágenes.
      - labels (list or None): Lista de etiquetas para las imágenes.
        Si es None, no se muestran etiquetas.
      - images_per_row (int): Número de imágenes por fila.
      - size (float): Tamaño base de las imágenes en la cuadrícula.
      - vertical_spacing (float): Espaciado vertical entre filas.
      - horizontal_spacing (float): Espaciado horizontal entre columnas.
    """
    images = list(images) if isinstance(images, np.ndarray) else images
    labels = list(labels) if isinstance(labels, np.ndarray) else labels

    total_images = len(images)
    num_rows = math.ceil(total_images / images_per_row)

    # Crear la figura y los ejes
    fig, axs = plt.subplots(
        num_rows, images_per_row,
        figsize=(images_per_row * size * fig_aspect[0], 
                 num_rows * size * fig_aspect[1])
    )

    # Asegurar que axs sea un array 1D
    if num_rows == 1:
        axs = axs if isinstance(axs, np.ndarray) else np.array([axs])
    axs = axs.flatten()

    for i in range(len(axs)):
        if i < total_images:
            plot_image(images[i], labels[i] if labels else None, axs[i])
        else:
            axs[i].axis('off')  # Ocultar ejes vacíos

    plt.subplots_adjust(hspace=vertical_spacing, wspace=horizontal_spacing)
    plt.show()

#-------------------------------------------------------------------------------------------

# Código de referencia: https://plotly.com/python/parallel-categories-diagram/
# Artículo de referencia: https://kosara.net/papers/2005/Bendix-InfoVis-2005.pdf
def plot_parallel_categories(df, cols_name, height=800, width=1200):
    """
    Crea un gráfico de categorías paralelas utilizando la biblioteca Plotly Express.

    Parameters:
        df (pd.DataFrame): DataFrame que contiene los datos.
        cols_name (list): Lista de nombres de las columnas categóricas que se incluirán en el gráfico de 
            categorías paralelas.
        title (str, opcional): Título del gráfico. Si no se proporciona, se generará un título por defecto 
            utilizando los nombres de las características.
        height (int, opcional): Altura de la figura en píxeles. Valor por defecto es 800.
        width (int, opcional): Ancho de la figura en píxeles. Valor por defecto es 1200.
    """

    # Build parcats dimensions
    categorical_dimensions = cols_name
    dimensions = [dict(values=df[label], label=label) for label in categorical_dimensions]

    # Create a colorscale
    color = np.zeros(len(df), dtype='uint8')
    colorscale = [[0, '#00CCBF'], [1, 'firebrick']]

    # Build figure as FigureWidget
    fig = go.FigureWidget(
            data=[go.Parcats(
                    domain={'y': [0, 0.9]}, dimensions=dimensions,
                    labelfont={'size': 16}, tickfont={'size': 13},
                    line={'colorscale': colorscale, 'cmin': 0, 'cmax': 1,
                          'color': color, 'shape': 'hspline'},
                    arrangement='freeform')
                 ]
          )

    if title is None:
        title = "Categorías paralelas de las características " + \
                ", ".join(f"{feature}" for feature in cols_name)

    # Update layout
    fig.update_layout(
        width=width, height=height,
        dragmode='lasso', hovermode='closest',
        margin=dict(l=200, r=200, t=70, b=60),
        title=title
    )

    # Update color callback
    def update_color(trace, points, state):
        # Update parcats colors
        new_color = np.zeros(len(df), dtype='uint8')
        new_color[points.point_inds] = 1
        fig.data[0].line.color = new_color

    # Register callback on parcats click
    fig.data[0].on_selection(update_color)

    plotly.offline.plot(fig, filename="graph.html")
    # Muestra el gráfico
    fig.show()
    

#-------------------------------------------------------------------------------------------------------------


def plot_interval_predictions(
    pred_point_values : torch.Tensor,
    pred_lower_bound : torch.Tensor,
    pred_upper_bound : torch.Tensor,
    true_values : torch.Tensor,
    figsize: Tuple[float, float] = (10, 6),
    dpi=150
) -> None:
    
    # Convierte los tensores a NumPy
    y_true = true_values.numpy()
    y_pred = pred_point_values.numpy()
    y_lower = pred_lower_bound.numpy()
    y_upper = pred_upper_bound.numpy()
    
    # Ordena por los valores predichos
    order = np.argsort(y_true)
    y_true = y_true[order]
    y_pred = y_pred[order]
    y_lower = y_lower[order]
    y_upper = y_upper[order]
    
    x = np.arange(len(y_true))
    
    # Grafica
    plt.figure(figsize=figsize, dpi=dpi)
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
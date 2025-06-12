import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.graph_objects as go
import math


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
        
#-------------------------------------------------------------------------------------------------------------

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
    
#-------------------------------------------------------------------------------------------------------------

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
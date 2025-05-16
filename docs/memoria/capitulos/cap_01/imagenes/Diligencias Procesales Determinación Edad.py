import pandas as pd
import matplotlib.pyplot as plt

# Menores Extranjeros No Acompañados y Diligencias procesales estimación de edad 
df = pd.DataFrame({
    'year' : [2011, 2012, 2013, 2014, 2015, 2016, 2017,  2018, 2019, 2020, 2021, 2022, 2023],
    'MENAs': [ 357,  275,  159,  223,  414,  588, 2345,  7026, 2873, 3307, 3048, 2375, 4865],
    'DPDE' : [2418, 1973, 1732, 2043, 2539, 2971, 5600, 12152, 7745, 4981, 6677, 4805, 7422]
})

# Crear la figura con tamaño personalizado (más alto que ancho)
fig, ax = plt.subplots(figsize=(8, 4))

# Graficar ambas líneas
ax.plot(df['year'], df['DPDE'], marker='o', label='DPDE', color='#3A59D1')

# Ajustes estéticos
# ax.set_title('Evolución del número de Diligencias Preprocesales de Determinación de Edad (2008–2023)', fontsize=14, pad=30)
ax.set_ylabel('Cantidad', fontsize=12)

# Establecer el rango del eje Y
ax.set_ylim(-100, 13000)

# Grid solo horizontal
ax.grid(False)
ax.yaxis.grid(True)

# Eliminar marcas (ticks) de los ejes X e Y
ax.tick_params(axis='x', length=0)
ax.tick_params(axis='y', length=0)

# Resaltar ejes X e Y
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Mostrar todas las fechas en X y rotarlas 90 grados
ax.set_xticks(df['year'])
plt.xticks(rotation=90, fontsize=12)
plt.yticks(fontsize=12)

plt.tight_layout()
plt.savefig('01_dpde_España.png', dpi=300, bbox_inches='tight')  # Guarda la imagen
plt.show()

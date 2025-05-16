import torch

print(f"Torch version: {torch.__version__}")
print(f"CUDA version: {torch.version.cuda}")

# Verifica si hay GPU disponible
if torch.cuda.is_available():
    print("CUDA está disponible.")
    print(f"Nombre de la GPU: {torch.cuda.get_device_name(0)}")
    print(f"Número de GPUs disponibles: {torch.cuda.device_count()}")
else:
    print("CUDA no está disponible. PyTorch no reconoce la GPU.")

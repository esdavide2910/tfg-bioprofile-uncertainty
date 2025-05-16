#!/bin/bash

#SBATCH --job-name prueba_pytorch
#SBATCH --partition dios
#SBATCH --mem=2G
#SBATCH --cpus-per-task 1
#SBATCH --gres=gpu:1
#SBATCH --output=output.txt
#SBATCH --error=error_report.err

# Fuerza la ejecución desde el directorio del proyecto
PROJECT_DIR="/mnt/homeGPU/dgonzalez/tfg-bioprofile-uncertainty"  # Ruta absoluta a tu proyecto
cd "$PROJECT_DIR" || { echo "Error: No se pudo acceder a $PROJECT_DIR"; exit 1; }

# Configuración de Conda
export PATH="/mnt/homeGPU/dgonzalez/conda_envs/pytorch_env2/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate /mnt/homeGPU/dgonzalez/conda_envs/envs/pytorch_env2

# Ejecución del script Python
python src/prueba_pytorch.py

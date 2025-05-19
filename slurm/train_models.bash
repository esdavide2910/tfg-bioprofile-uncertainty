#!/bin/bash

#SBATCH --job-name train_models
#SBATCH --partition dios
##SBATCH --nodelist=
#SBATCH --exclude=titan,zeus
#SBATCH --mem=20G
#SBATCH --cpus-per-task 1
#SBATCH --gres=gpu:1
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Fuerza la ejecución desde el directorio del proyecto
PROJECT_DIR="/mnt/homeGPU/dgonzalez/tfg-bioprofile-uncertainty"  # Ruta absoluta a tu proyecto
cd "$PROJECT_DIR" || { echo "Error: No se pudo acceder a $PROJECT_DIR"; exit 1; }

# Configuración de Conda
export PATH="/mnt/homeGPU/dgonzalez/conda_envs/pytorch_env2/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate /mnt/homeGPU/dgonzalez/conda_envs/envs/pytorch_env2

# Ejecución del script Python
python src/model1.py > results/AE_maxillofacial/report_train_model1.txt 2>&1
#!/bin/bash

#SBATCH --job-name train_models
#SBATCH --partition dios
##SBATCH --nodelist=dionisio
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
python src/modelA.py > results/AE_maxillofacial/report_train_modelA.txt 2>&1
# python src/modelB.py > results/AE_maxillofacial/report_train_modelB.txt 2>&1
# python src/modelC.py > results/AE_maxillofacial/report_train_modelC.txt 2>&1
# python src/modelD.py > results/AE_maxillofacial/report_train_modelD.txt 2>&1
#!/bin/bash

#SBATCH --job-name train_models
#SBATCH --partition dios
#SBATCH --exclude=titan,zeus
#SBATCH --mem=20G
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
#SBATCH --gres=gpu:1
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Fuerza la ejecución desde el directorio del proyecto
PROJECT_DIR="/mnt/homeGPU/dgonzalez/tfg-bioprofile-uncertainty"
cd "$PROJECT_DIR" || { echo "Error: No se pudo acceder a $PROJECT_DIR"; exit 1; }

# Configuración de Conda
export PATH="/mnt/homeGPU/dgonzalez/conda_envs/pytorch_env/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate /mnt/homeGPU/dgonzalez/conda_envs/envs/pytorch_env


# Ejecución de los scripts Python

for iteration in {1..1}; do

    python src/AMSC_maxillofacial.py --pred_model_type base \
        --test \
        --load_model_path models/AMSC_model01_base.pth \
        --output_stream results/AMSC_model01_base_report.txt --append_output \
        --ignore_warnings 

    python src/AMSC_maxillofacial.py --pred_model_type LAC --confidence 0.9 \
        --calibrate --test \
        --load_model_path models/AMSC_model02_LAC.pth \
        --save_model_path models/AMSC_model02_LAC.pth \
        --output_stream results/AMSC_model02_LAC_report.txt --append_output \
        --ignore_warnings 

    python src/AMSC_maxillofacial.py --pred_model_type MCM --confidence 0.9 \
        --calibrate --test \
        --load_model_path models/AMSC_model03_LAC.pth \
        --save_model_path models/AMSC_model03_MCM.pth \
        --output_stream results/AMSC_model03_MCM_report.txt --append_output \
        --ignore_warnings 

    python src/AMSC_maxillofacial.py --pred_model_type APS --confidence 0.9 \
        --calibrate --test \
        --load_model_path models/AMSC_model04_LAC.pth \
        --save_model_path models/AMSC_model04_APS.pth \
        --output_stream results/AMSC_model04_APS_report.txt --append_output \
        --ignore_warnings 

done
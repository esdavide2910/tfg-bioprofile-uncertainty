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
PROJECT_DIR="/mnt/homeGPU/dgonzalez/tfg-bioprofile-uncertainty"  # Ruta absoluta a tu proyecto
cd "$PROJECT_DIR" || { echo "Error: No se pudo acceder a $PROJECT_DIR"; exit 1; }

# Configuración de Conda
export PATH="/mnt/homeGPU/dgonzalez/conda_envs/pytorch_env/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate /mnt/homeGPU/dgonzalez/conda_envs/envs/pytorch_env


# Ejecución de los scripts Python

for iteration in {1..1}; do

    python src/AMM_maxillofacial.py --pred_model_type base \
        --test \
        --load_model_path models/AMM_model01_base.pth \
        --output_stream results/AMM_model01_base_report.txt --append_output \
        --save_test_results results/AMM_maxillofacial_test_results.csv -i $iteration \
        --ignore_warnings 

    python src/AMM_maxillofacial.py --pred_model_type LAC --confidence 0.9 \
        --calibrate --test \
        --load_model_path models/AMM_model02_LAC.pth \
        --save_model_path models/AMM_model02_LAC.pth \
        --output_stream results/AMM_model02_LAC_report.txt --append_output \
        --save_test_results results/AMM_maxillofacial_test_results.csv -i $iteration \
        --ignore_warnings 

    python src/AMM_maxillofacial.py --pred_model_type LAC --confidence 0.95 \
        --calibrate --test \
        --load_model_path models/AMM_model02_LAC.pth \
        --save_model_path models/AMM_model02_LAC.pth \
        --output_stream results/AMM_model02_LAC_report.txt --append_output \
        --save_test_results results/AMM_maxillofacial_test_results.csv -i $iteration \
        --ignore_warnings 

    python src/AMM_maxillofacial.py --pred_model_type MCM --confidence 0.9 \
        --calibrate --test \
        --load_model_path models/AMM_model03_MCM.pth \
        --save_model_path models/AMM_model03_MCM.pth \
        --output_stream results/AMM_model03_MCM_report.txt --append_output \
        --save_test_results results/AMM_maxillofacial_test_results.csv -i $iteration \
        --ignore_warnings 

    python src/AMM_maxillofacial.py --pred_model_type MCM --confidence 0.95 \
        --calibrate --test \
        --load_model_path models/AMM_model03_MCM.pth \
        --save_model_path models/AMM_model03_MCM.pth \
        --output_stream results/AMM_model03_MCM_report.txt --append_output \
        --save_test_results results/AMM_maxillofacial_test_results.csv -i $iteration \
        --ignore_warnings 

done

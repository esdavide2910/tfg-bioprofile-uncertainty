#!/bin/bash

#SBATCH --job-name train_models
#SBATCH --partition dios
#SBATCH --exclude=titan,zeus,atenea
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


for iteration in {10..10}; do

    python src/AMSC_maxillofacial.py --pred_model_type LAC --confidence 0.95 \
        --train --calibrate --test \
        --load_model_path models/AE_model02_ICP.pth \
        --save_model_path models/AMSC_model02_LAC.pth \
        --output_stream results/AMSC_model02_LAC_report.txt --append_output \
        --save_test_results results/AMSC_maxillofacial_test_results.csv -i $iteration \
        --ignore_warnings

    python src/AMSC_maxillofacial.py --pred_model_type MCM --confidence 0.95 \
        --calibrate --test \
        --load_model_path models/AE_model02_ICP.pth \
        --save_model_path models/AMSC_model03_MCM.pth \
        --output_stream results/AMSC_model03_MCM_report.txt --append_output \
        --save_test_results results/AMSC_maxillofacial_test_results.csv -i $iteration \
        --ignore_warnings

    python src/AMSC_maxillofacial.py --pred_model_type APS --confidence 0.95 \
        --calibrate --test \
        --load_model_path models/AMSC_model02_LAC.pth \
        --save_model_path models/AMSC_model04_APS.pth \
        --output_stream results/AMSC_model04_APS_report.txt --append_output \
        --save_test_results results/AMSC_maxillofacial_test_results.csv -i $iteration \
        --ignore_warnings

    python src/AMSC_maxillofacial.py --pred_model_type RAPS --confidence 0.95 \
        --calibrate --test \
        --load_model_path models/AMSC_model02_LAC.pth \
        --save_model_path models/AMSC_model05_RAPS.pth \
        --output_stream results/AMSC_model05_RAPS_report.txt --append_output \
        --save_test_results results/AMSC_maxillofacial_test_results.csv -i $iteration \
        --ignore_warnings

    python src/AMSC_maxillofacial.py --pred_model_type SAPS --confidence 0.95 \
        --calibrate --test \
        --load_model_path models/AMSC_model02_LAC.pth \
        --save_model_path models/AMSC_model06_SAPS.pth \
        --output_stream results/AMSC_model06_SAPS_report.txt --append_output \
        --save_test_results results/AMSC_maxillofacial_test_results.csv -i $iteration \
        --ignore_warnings

done


# srun --partition dios --exclude=titan,zeus --mem=10G --ntasks 1 --cpus-per-task 1 --gres=gpu:1 \

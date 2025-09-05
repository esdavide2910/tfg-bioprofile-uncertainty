#!/bin/bash

#SBATCH --job-name test_MC
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

for iteration in {1..10}; do

    # Precarga los pesos del modelo base de AE y ajusta el head
    python src/II_MC_maxillofacial.py --pred_method base \
        --test \
        --load_model_path models/II_MC_models/II_MC_model01_base_${iteration}.pth \
        --save_model_path models/II_MC_models/II_MC_model01_base_${iteration}.pth \
        --output_stream results/II_MC_maxillofacial_report.txt --append_output \
        --save_test_results results/II_MC_maxillofacial_test_results.csv -i $iteration \
        --ignore_warnings 

    # Precarga los pesos del modelo ICP de AE y ajusta el head
    python src/II_MC_maxillofacial.py --pred_method LAC --confidence 0.95 \
        --calibrate --test \
        --load_model_path models/II_MC_models/II_MC_model02_CP_${iteration}.pth \
        --save_model_path models/II_MC_models/II_MC_model02_CP_${iteration}.pth \
        --output_stream results/II_MC_maxillofacial_report.txt --append_output \
        --save_test_results results/II_MC_maxillofacial_test_results.csv -i $iteration \
        --ignore_warnings 

    # Carga directamente el modelo LAC
    python src/II_MC_maxillofacial.py --pred_method MCM --confidence 0.95 \
        --calibrate --test \
        --load_model_path models/II_MC_models/II_MC_model02_CP_${iteration}.pth \
        --save_model_path models/II_MC_models/II_MC_model02_CP_${iteration}.pth \
        --output_stream results/II_MC_maxillofacial_report.txt --append_output \
        --save_test_results results/II_MC_maxillofacial_test_results.csv -i $iteration \
        --ignore_warnings 

done

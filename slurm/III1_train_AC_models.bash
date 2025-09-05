#!/bin/bash

#SBATCH --job-name train_AC
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

for iteration in {1..10}; do

    python src/III_AC_maxillofacial.py --pred_method base \
        --train_head --test \
        --load_model_path models/I_AE_models/I_AE_model01_base_${iteration}.pth \
        --save_model_path models/III_AC_models/III_AC_model01_base_${iteration}.pth \
        --output_stream results/III_AC_maxillofacial_report.txt --append_output \
        --save_test_results results/III_AC_maxillofacial_test_results.csv -i $iteration \
        --ignore_warnings 

    python src/III_AC_maxillofacial.py --pred_method LAC --confidence 0.95 \
        --train_head --calibrate --test \
        --load_model_path models/I_AE_models/I_AE_model02_ICP_${iteration}.pth \
        --save_model_path models/III_AC_models/III_AC_model02_CP_${iteration}.pth \
        --output_stream results/III_AC_maxillofacial_report.txt --append_output \
        --save_test_results results/III_AC_maxillofacial_test_results.csv -i $iteration \
        --ignore_warnings 

    python src/III_AC_maxillofacial.py --pred_method MCM --confidence 0.95 \
        --calibrate --test \
        --load_model_path models/III_AC_models/III_AC_model02_CP_${iteration}.pth \
        --save_model_path models/III_AC_models/III_AC_model02_CP_${iteration}.pth \
        --output_stream results/III_AC_maxillofacial_report.txt --append_output \
        --save_test_results results/III_AC_maxillofacial_test_results.csv -i $iteration \
        --ignore_warnings 

    python src/III_AC_maxillofacial.py --pred_method APS --confidence 0.95 \
        --calibrate --test \
        --load_model_path models/III_AC_models/III_AC_model02_CP_${iteration}.pth \
        --save_model_path models/III_AC_models/III_AC_model02_CP_${iteration}.pth \
        --output_stream results/III_AC_maxillofacial_report.txt --append_output \
        --save_test_results results/III_AC_maxillofacial_test_results.csv -i $iteration \
        --ignore_warnings 

    python src/III_AC_maxillofacial.py --pred_method RAPS --confidence 0.95 \
        --calibrate --test \
        --load_model_path models/III_AC_models/III_AC_model02_CP_${iteration}.pth \
        --save_model_path models/III_AC_models/III_AC_model02_CP_${iteration}.pth \
        --output_stream results/III_AC_maxillofacial_report.txt --append_output \
        --save_test_results results/III_AC_maxillofacial_test_results.csv -i $iteration \
        --ignore_warnings 

    python src/III_AC_maxillofacial.py --pred_method SAPS --confidence 0.95 \
        --calibrate --test \
        --load_model_path models/III_AC_models/III_AC_model02_CP_${iteration}.pth \
        --save_model_path models/III_AC_models/III_AC_model02_CP_${iteration}.pth \
        --output_stream results/III_AC_maxillofacial_report.txt --append_output \
        --save_test_results results/III_AC_maxillofacial_test_results.csv -i $iteration \
        --ignore_warnings 

done
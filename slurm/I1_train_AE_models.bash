#!/bin/bash

#SBATCH --job-name train_AE
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

for iteration in {2..10}; do

    # python src/I_AE_maxillofacial.py --pred_method base \
    #     --train --test \
    #     --save_model_path models/I_AE_models/I_AE_model01_base_${iteration}.pth \
    #     --output_stream results/I_AE_maxillofacial_report.txt --append_output \
    #     --save_test_results results/I_AE_maxillofacial_test_results.csv -i $iteration \
    #     --ignore_warnings 

    # python src/I_AE_maxillofacial.py --pred_method ICP --confidence 0.95 \
    #     --train --calibrate --test \
    #     --save_model_path models/I_AE_models/I_AE_model02_ICP_${iteration}.pth \
    #     --output_stream results/I_AE_maxillofacial_report.txt --append_output \
    #     --save_test_results results/I_AE_maxillofacial_test_results.csv -i $iteration \
    #     --ignore_warnings 

    # python src/I_AE_maxillofacial.py --pred_method QR --confidence 0.95 \
    #     --train --test \
    #     --save_model_path models/I_AE_models/I_AE_model03_QR_${iteration}.pth \
    #     --output_stream results/I_AE_maxillofacial_report.txt --append_output \
    #     --save_test_results results/I_AE_maxillofacial_test_results.csv -i $iteration \
    #     --ignore_warnings 

    python src/I_AE_maxillofacial.py --pred_method CQR --confidence 0.95 \
        --train --calibrate --test \
        --save_model_path models/I_AE_models/I_AE_model04_CQR_${iteration}.pth \
        --output_stream results/I_AE_maxillofacial_report.txt --append_output \
        --save_test_results results/I_AE_maxillofacial_test_results.csv -i $iteration \
        --ignore_warnings 
        
done

#--------

# srun --partition dios --exclude=titan,zeus --mem=20G --ntasks 1 --cpus-per-task 4 --gres=gpu:1 \
#     python src/I_AE_maxillofacial.py --pred_method base \
#         --train --test \
#         --save_model_path models/I_AE_models/I_AE_model01_base_1.pth \
#         --output_stream results/I_AE_maxillofacial_report.txt --append_output \
#         --save_test_results results/I_AE_maxillofacial_test_results.csv -i 1 \
#         --ignore_warnings 




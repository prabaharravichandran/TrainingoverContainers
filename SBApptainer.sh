#!/bin/bash -l
#SBATCH --job-name=Apptainer
#SBATCH --output=Apptainer.out
#SBATCH --no-requeue
#SBATCH --partition=gpu_a100
#SBATCH --account=aafc_phenocart__gpu_a100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=128000M  # Request 128 GB per CPU
#SBATCH --gres=gpu:4
#SBATCH --qos=low
#SBATCH --time=48:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=prabahar.ravichandran@agr.gc.ca

# Load CUDA module (or set up environment)
. ssmuse-sh -d /fs/ssm/hpco/exp/cuda-12.4.1

# Change to the working directory
cd /gpfs/fs7/aafc/phenocart/PhenomicsProjects/TrainingoverContainers/Images

# Execute the Python script inside the container
apptainer exec \
   --nv \
   --fakeroot \
   --bind /gpfs/fs7/aafc/phenocart:/mnt:rw \
   --contain \
   --no-home \
   trainingcontainer_sandbox/ \
   bash -c "export CUDA_HOME=/usr/local/cuda; \
            export PATH=\$CUDA_HOME/bin:\$PATH; \
            export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\$LD_LIBRARY_PATH; \
            export LIBDEVICE_DIR=/usr/local/cuda/nvvm/libdevice; \
            source /home/venv/bin/activate; \
            python /home/ubuntu/scripts/yestimation_branch.py"

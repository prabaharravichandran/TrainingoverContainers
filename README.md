# Building containers for training deep learning models at High-Performance Clusters(HPCs)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Building containers

sbatch make.sbatch, recipe is in Singularity.def; python requirements are in requirements.txt

```bash
sbatch make.sbatch
squeue -u prr000
```

## Training at GPSC

Testing MongoDB
```bash
mongod --dbpath=/mnt/IDEs/MongoDB/data --logpath=/tmp/mongod.log --fork
mongod --fork --config="/home/ubuntu/mongod.config"
```

```text
about to fork child process, waiting until server is ready for connections.
forked process: 921
child process started successfully, parent exiting
```

We can also add the commands to activate the environment and run the script runscript ".singularity.d/runscript" and covert into .sif

```bash
singularity build trainingcontainer_sandbox.sif trainingcontainer_sandbox/
```

salloc - allocate resources for a job in HPC

```bash
salloc --account=aafc_phenocart__gpu_a100 \
 --partition=gpu_a100 \
 --time=12:00:00 \
 --nodes=1 \
 --cpus-per-task=4 \
 --mem-per-cpu=64000M \
 --gpus=4 \
 --qos=low
```

Use srun to launch tasks within an allocated job
```bash
squeue -u prr000
srun --job-id=xxxxxxx --pty bash
```

From sandbox,

```bash
. ssmuse-sh -d /fs/ssm/hpco/exp/cuda-12.4.1
apptainer shell \
   --nv \
   --fakeroot \
   --bind /gpfs/fs7/aafc/phenocart:/mnt \
   --contain \
   --no-home \
   trainingcontainer_sandbox/
```

writable

```bash
. ssmuse-sh -d /fs/ssm/hpco/exp/cuda-12.4.1
apptainer shell \
   --writable \
   --fakeroot \
   --bind /gpfs/fs7/aafc/phenocart:/mnt:rw \
   --contain \
   --no-home \
   trainingcontainer_sandbox/
```

```bash
. ssmuse-sh -d /fs/ssm/hpco/exp/cuda-12.4.1
apptainer shell \
   --nv \
   --fakeroot \
   --bind /gpfs/fs7/aafc/phenocart:/mnt \
   --contain \
   --no-home \
   trainingcontainer_sandbox.sif
```

From sif, sifs are usually faster;

```bash
. /home/venv/bin/activate
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```
```testOutput
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LIBDEVICE_DIR=/usr/local/cuda/nvvm/libdevice
. /home/venv/bin/activate
python /home/ubuntu/scripts/fandmestimation.py
```
Submitting this job using slurm,

```bash
#!/bin/bash -l
#SBATCH --job-name=Process
#SBATCH --output=Process.out
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

```

```bash
rsync --recursive --progress --stats --checksum -e "ssh -i /home/prr000/.ssh/hybridcloud2025_GPSC" ufps 3.98.237.27:/fs/phenocart-app/prr000/Projects/Deployment
```

lscpu > cpu_static_info.txt
cat /proc/cpuinfo >> cpu_static_info.txt

salloc

```bash
salloc \
  --job-name=Process \
  --partition=gpu_a100 \
  --account=aafc_phenocart__gpu_a100 \
  --nodes=1 \
  --mem=128G \
  --gpus=4 \
  --qos=low \
  --time=4:00:00
```
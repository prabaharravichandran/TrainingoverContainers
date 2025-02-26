# Building containers for training deep learning models

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
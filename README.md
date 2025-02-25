# Building containers for training deep learning models

## Building containers

sbatch make.sbatch, recipe is in Singularity.def; python requirements are in requirements.txt

```bash
sbatch make.sbatch
squeue -u prr000
```

## Training at GPSC

salloc - allocate resources for a job in HPC

```bash
salloc --account=aafc_phenocart__gpu_a100 \
 --partition=gpu_a100 \
 --time=01:00:00 \
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

Testing MongoDB
```bash
mongod --dbpath=/mnt/IDEs/MongoDB/data --logpath=/tmp/mongod.log --fork
mongod --fork --config="/home/ubuntu/mongod.conf"
```

```text
about to fork child process, waiting until server is ready for connections.
forked process: 921
child process started successfully, parent exiting
```

```bash
. /home/venv/bin/activate
python /home/ubuntu/scripts/yieldestimation.py
```

We can also add the commands to activate the environment and run the script runscript ".singularity.d/runscript" and covert into .sif

```bash
singularity build trainingcontainer_sandbox.sif trainingcontainer_sandbox/
```
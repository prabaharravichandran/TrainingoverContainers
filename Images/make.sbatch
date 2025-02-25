#!/bin/bash
#SBATCH --account=aafc_phenocart__gpu_a100
#SBATCH --partition=gpu_a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=124G  # 124 GB total memory
#SBATCH --comment="tmpfs_size=90%"
#SBATCH --time=1:00:00
#SBATCH --qos=low

set -ex

# Proxy settings for web access
export https_proxy=http://webproxy.science.gc.ca:8888
export http_proxy=http://webproxy.science.gc.ca:8888
export no_proxy=localhost,science.gc.ca

# Define Singularity temporary directories
export APPTAINER_TMPDIR=/tmp/singularity/tmp
export APPTAINER_CACHEDIR=/tmp/singularity/cache

# Ensure TMP and CACHE directories exist
mkdir -p "${APPTAINER_TMPDIR}"
mkdir -p "${APPTAINER_CACHEDIR}"

# Unset problematic variables
unset SINGULARITY_BINDPATH

# Run Makefile with multiple cores (if possible)
make -j$(nproc)

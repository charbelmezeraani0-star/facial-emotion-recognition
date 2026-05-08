#!/bin/bash
NVIDIA_LIB=$(python3 -c "import site; print(site.getusersitepackages())")/nvidia
export LD_LIBRARY_PATH="$NVIDIA_LIB/cuda_runtime/lib:$NVIDIA_LIB/cublas/lib:$NVIDIA_LIB/cudnn/lib:$NVIDIA_LIB/cufft/lib:$NVIDIA_LIB/curand/lib:$NVIDIA_LIB/cusolver/lib:$NVIDIA_LIB/cusparse/lib:$NVIDIA_LIB/nvjitlink/lib:$LD_LIBRARY_PATH"

# Add ptxas (CUDA PTX compiler) and libdevice to path so XLA/Triton can compile GPU kernels
export PATH="$NVIDIA_LIB/cuda_nvcc/bin:$PATH"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$NVIDIA_LIB/cuda_nvcc"

cd "/home/charbel-mezeraani/cv project"
python3 -u training/train.py "$@"

# HENKES_GAN

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.xxx.svg)](https://doi.org/10.5281/zenodo.xxx)

Code of the publication "xxx" published in 
https://doi.org/xxx by 
Alexander Henkes and Henning Wessels from TU Braunschweig.

Please cite the following paper:

    @article{xxx,
      title={xxx},
      author={Henkes, Alexander and Wessels, Henning},
      journal={xxx},
      volume={xxx},
      pages={xxx},
      year={xxx},
      publisher={Elsevier}
    }

... and the code using the CITATION.cff file.

# Requirements
The requirements can be found in
    
    requirements.txt

and may be installed via pip:

    pip install -r requirements.txt

# Docker image
You can download a pre-built Docker image via:

    docker pull ahenkes1/xxx:1.0.0

If you want to build the Docker image, the official TensorFlow image is needed:

    https://www.tensorflow.org/install/docker

Build via

    docker build -f ./Dockerfile --pull -t ahenkes1/xxx:1.0.0 .

Execute via

    docker run --gpus all -it -v YOUR_LOCAL_OUTPUT_FOLDER:/home/docker_user/src/save_files/ --rm ahenkes1/xxx:1.0.0 --help

where 'YOUR_LOCAL_OUTPUT_FOLDER' is an absolute path to a directory on your 
system. This will show the help.

Execute the code using standard parameters as

    docker run --gpus all -it -v YOUR_LOCAL_OUTPUT_FOLDER:/home/docker_user/src/save_files --rm ahenkes1/xxx:1.0.0 

# Using XLA
The code may run using XLA (faster) using the following flag:

    XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda-11.2 python3 main.py --help

where the correct cuda path and version have to be used.
The Docker image runs XLA natively.

# GPU
The code uses mixed-precision. If your GPU has TensorCores, it will run much 
faster. Otherwise, a warning will be displayed. Nevertheless, the memory 
consumption is much lower in either case.

# Tensorboard
The code logs several metrics during training, which can be accessed via 
Tensorboard. The logs can be found in the corresponding output folders.
    
    https://www.tensorflow.org/tensorboard

<!---
Copyright (c) 2023 Advanced Micro Devices, Inc. (AMD)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
--->
# Creating a PyTorch/TensorFlow Code Environment on AMD GPUs

**Goal**: The machine learning ecosystem is quickly exploding and we aim to make porting to AMD GPUs simple with this series of machine learning blogposts.

**Audience**: Data scientists and machine learning practitioners, as well as software engineers who use PyTorch/TensorFlow on AMD GPUs. You can be new to machine learning, or experienced in using Nvidia GPUs. 

**Motivation**: Because when starting a new machine learning project, you may notice that many existing codes on GitHub are almost always CUDA based. If you have AMD GPUs and follow their instructions on running the code, it often does not work. We provide steps, based on our experience, that can help you get a code environment working for your experiments and to manage working with CUDA-based code repositories on AMD GPUs.

**Differentiator** from existing online resources:
- This is from a machine learning practitioner's perspective, to guide you away from rabbit holes due to habits and preferences, such as using Jupyter Notebooks and pip install. 
- This is *not* to teach you how to install PyTorch/TensorFlow on ROCm because this step alone often times cannot lead to successfully running machine learning code.
- This is *not* to teach you how to HIPify code, but instead, to let you know that sometimes you don't even need that step.
- As of today, this is the only documentation so far on the internet that has end-to-end instructions on how to create PyTorch/TensorFlow code environment on AMD GPUs. 

The prerequisite is to have ROCm installed, follow the instructions [here](https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-rocm-installation-readme/) and [here](https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html).

## Install PyTorch or TensorFlow on ROCm

### Option 1. PyTorch 

We recommend following the instructions on the [official ROCm PyTorch website](https://rocm.docs.amd.com/en/latest/how_to/pytorch_install/pytorch_install.html).

### Option 2. TensorFlow 

We recommend following the instructions on the [official ROCm TensorFlow website](https://rocm.docs.amd.com/en/latest/how_to/tensorflow_install/tensorflow_install.html). 

*Note*: We also strongly recommend using Docker image with [PyTorch](https://rocm.docs.amd.com/en/latest/how_to/pytorch_install/pytorch_install.html#option-1-recommended-use-docker-image-with-pytorch-pre-installed) or [TensorFlow](https://rocm.docs.amd.com/en/latest/how_to/tensorflow_install/tensorflow_install.html#option-1-install-tensorflow-using-docker-image) pre-installed. The reason is that if you create a virtual environment or conda environment, certain ROCm dependencies may not be properly installed. It can be non-trivial to install dependencies.  

*Note*: You don’t need flag “--gpus all” to run docker on AMD GPUs.  

## Git clone the source code you want to run

```bash
git clone –-recursive <https://github.com/project/repo.git>
```

## Install library requirements based on the GitHub repository 

- Skip the commands that create virtual environments or conda environments. They are usually in `machine_install.sh` or `setup.sh` files. 
- Go directly to the library list and remove `torch` and `tensorflow` since these are CUDA-based by default. The docker containers should already have those libraries installed for ROCm. You can usually find the library list in requirements.txt.  
- Run `pip3 install –r requirements.txt` where `requirements.txt` contains single lines with package names (and possibly package versions).
  
## Run your code

If you can run your code without problems, then you have successfully created a code environment on AMD GPUs! 

If not, then it may be due to the additional packages in `requirements.txt` depending on CUDA, which needs to be HIPified to run on AMD GPUs.

## Obtain HIPified library source code  

### Option 1. Find existing HIPified library source code

You can simply search online or on GitHub for "library_name" + "ROCm". The HIPified code will pop up if it exists. 

Since this step is not trivial, here is an example:

If you are trying to run large language model related code, you may need the library `bitsandbytes` (see [link](https://github.com/TimDettmers/bitsandbytes)). 

Searching online for "bitsandbytes ROCm" you will find [this fork](https://github.com/agrocylo/bitsandbytes-rocm) which adds ROCm support with a HIP compilation target. 

```bash
git clone https://github.com/agrocylo/bitsandbytes-rocm 
cd bitsandbytes-rocm 
export ROCM_HOME=/opt/rocm/ 
make hip -j 
python3 setup.py install 
```

Note: the installation location may have the version number such as */opt/rocm-5.5.0*.

### Option 2. HIPify code if necessary 

We recommend following the below tutorials for this option.  

- https://rocm.docs.amd.com/projects/HIPIFY/en/latest/
- https://enccs.github.io/amd-rocm-development/porting_hip/
- [Tutorial on Porting CUDA to HIP](https://www.youtube.com/watch?v=57FwfePRd-Y)  
- https://www.admin-magazine.com/HPC/Articles/Porting-CUDA-to-HIP
  
## Commit changes to Docker Image

Once you finish modifying the new Docker container following the first step ("Install PyTorch or TensorFlow on ROCm"), exit out:

```bash
exit
```

Prompt the system to display a list of launched containers and find the docker container ID:

```bash
docker ps -a
```

Create a new image by committing the changes:

```bash
docker commit [CONTAINER_ID] [new_image_name]
```

In conclusion, this article introduces key steps on how to create PyTorch/TensorFlow code environment on AMD GPUs. ROCm is a maturing ecosystem and more GitHub codes will eventually contain ROCm/HIPified ports. Future posts to AMD lab notes will discuss the specifics of porting from CUDA to HIP, as well guides to running popular community models from [HuggingFace](https://huggingface.co/).

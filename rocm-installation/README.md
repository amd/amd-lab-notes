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
# AMD ROCm™ installation

AMD ROCm™ is the first open-source software development platform for HPC/Hyperscale-class GPU computing. AMD ROCm™ brings the UNIX philosophy of choice, minimalism and modular software development to GPU computing. Please see the AMD [Open Software Platform for GPU](https://www.amd.com/en/graphics/servers-solutions-rocm) and [ROCm Informational Portal](https://docs.amd.com/) pages for more information. 

Installation of the AMD ROCm™ software package can be challenging without a clear understanding of the pieces involved and the flow of the installation process. This introductory material shows how to install ROCm on a workstation with an AMD GPU card that supports the AMD GFX9 architecture. A follow on blog will discuss installing ROCm in other environments, such as a Docker Container, Linux LXC or a full HPC installation. 

The website [https://docs.amd.com](https://docs.amd.com) contains links to the Release, Support and API documentation for ROCm. Please refer to the [Installation Guide](https://docs.amd.com/bundle/ROCm-Installation-Guide-v5.4/page/Introduction_to_ROCm_Installation_Guide_for_Linux.html) and [Hardware / Software Support Guide](https://docs.amd.com/bundle/Hardware_and_Software_Reference_Guide/page/Hardware_and_Software_Support.html) for the software and hardware supported by the V 5.4 release of ROCm. This post will be based on an Ubuntu 20.04 operating system and the AMD MI50 GPU card. The full installation process is documented at [How to Install](https://docs.amd.com/bundle/ROCm-Installation-Guide-v5.4/page/How_to_Install_ROCm.html#_How_to_Install) page.

AMD ROCm™ is a brand name for the ROCm open software platform supporting GPUs using AMD's CDNA, and RDNA GPU architectures. The platform includes drivers and runtimes for libraries and developer tools.

 Three installation options will be described in this blog post:

 1. Installation of ROCm using an AMD provided script.
 2. Support for multiple ROCm versions on one system.
 3. Installation of ROCm using Ubuntu's apt-get.

## Option 1

AMD provides an installation script for specific operating system and ROCm versions. The script name and download location can be different for each combination of O/S and ROCm so check the [How to Install page](https://docs.amd.com/bundle/ROCm-Installation-Guide-v5.4/page/How_to_Install_ROCm.html#_How_to_Install) for your specific combination. We are using Ubuntu 20.04 and installing ROCm 5.4 and find that the script is named *amdgpu-install_5.4.50400-1_all*.

```bash
sudo apt-get update
wget https://repo.radeon.com/amdgpu-install/5.4/ubuntu/focal/amdgpu-install_5.4.50400-1_all.deb
sudo apt-get install ./amdgpu-install_5.4.50400-1_all.deb
```
Once the amdgpu-install script has been extracted,  it can be used to install the kernel code, libraries and developer code. For a typical HPC environment HIP, ROCm and the kernel drivers should be sufficient:
```bash
sudo amdgpu-install --usecase=hiplibsdk,rocm,dkms
```
Additional libraries can be installed and the list of available use cases can be found using:
```bash
sudo amdgpu-install --list-usecase
```
The ROCm code installs to /opt/rocm by default. You can verify that ROCm is installed by running
```bash
/opt/rocm/bin/rocminfo
```
and checking that the card was detected by the software. The supported GPU card will start with "GFX9". 

## Option 2

If ROCm is already installed, an old version can be removed and a new version may be installed. Alternatively, an additional version may be installed while keeping the old version. Installing an additional version creates a multi-version system and care must be used to ensure the proper paths are in the environment variables. For example, the current ```/opt/rocm``` may now be ```/opt/rocm-5.4.0``` or ```/opt/rocm-5.3.2``` depending on the installed ROCm versions. 

To remove all old versions use:
```bash
sudo amdgpu-uninstall --rocmrelease=all
```

AMD provides an installation script for specific operating system and ROCm versions. The script name and download location can be different for each combination of O/S and ROCm so check the [How to Install page](https://docs.amd.com/bundle/ROCm-Installation-Guide-v5.4/page/How_to_Install_ROCm.html#_How_to_Install) for your specific combination. We are using Ubuntu 20.04 and installing ROCm 5.4 and find that the script is named *amdgpu-install_5.4.50400-1_all*.

```bash
sudo apt-get update
wget https://repo.radeon.com/amdgpu-install/5.4/ubuntu/focal/amdgpu-install_5.4.50400-1_all.deb
sudo apt-get install ./amdgpu-install_5.4.50400-1_all.deb
```
Once the amdgpu-install script has been extracted,  it can be used to install the kernel code, libraries and developer code.
The steps below will install the kernel driver code at level 5.4.0 and the libraries at level 5.4.0, 5.3.2 and 5.2.3. For a typical HPC environment HIP and the ROCm libraries should be sufficient:
```bash
sudo amdgpu-install --usecase=hiplibsdk,rocm,dkms --rocmrelease=5.4.0
sudo amdgpu-install --usecase=hiplibsdk,rocm --rocmrelease=5.3.2 --no-dkms
sudo amdgpu-install --usecase=hiplibsdk,rocm --rocmrelease=5.2.3 --no-dkms
```
Additional libraries can be installed and the list of available use cases can be found using:
```bash
sudo amdgpu-install --list-usecase
```
The ROCm code installs to /opt/rocm by default. You can verify that ROCm is installed by running
```bash
/opt/rocm-5.4.0/bin/rocminfo
```
and checking that the card was detected by the software. The supported GPU card will start with "GFX9". 

## Option 3

While AMD provides an installation script for specific operating system and ROCm versions, the script will ultimately install the software using the O/S standard installation software. In the case of Ubuntu, the script will use *apt-get* to  install ROCm.
The full apt-get process is shown on the [How to Install page](https://docs.amd.com/bundle/ROCm-Installation-Guide-v5.4/page/How_to_Install_ROCm.html#_How_to_Install). The steps below will install ROCm 5.4 with a custom version of the apt-get commands. 

Determine the location of the ROCm software to install and HIP and the related ROCm software:
```bash
export ROCM_REPO_BASEURL="https://repo.radeon.com/rocm/apt/5.4/"
export ROCM_REPO_COMP="ubuntu"
export ROCM_REPO_BUILD="main"
echo "deb [arch=amd64 trusted=yes] ${ROCM_REPO_BASEURL} ${ROCM_REPO_COMP} ${ROCM_REPO_BUILD}" > /etc/apt/sources.list.d/rocm.list
sudo apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
 libdrm-amdgpu* \
 initramfs-tools \
 libtinfo* \
 initramfs-tools \
 rocm-llvm \
 rocm-hip-runtime \
 rocm-hip-sdk \
 roctracer-dev
```
To find a list of other parts of ROCm that can also be installed using apt-get try using apt-cache:
```bash
apt-cache search rocm
```
The ROCm code installs to /opt/rocm by default. You can verify that ROCm is installed by running
```bash
/opt/rocm/bin/rocminfo
```
and checking that the card was detected by the software. The supported GPU card will start with "GFX9". 

Note: It is not recommended to mix the apt-get and amdgpu-install methods. Pick one or the other approach for installing ROCm.

If you have any questions or comments, you can reach out to us on our [mailing list](mailto:dl.amd-lab-notes@amd.com).

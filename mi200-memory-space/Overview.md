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

# AMD Instinct™ MI200 GPU memory space overview

The HIP API supports a wide variety of allocation methods for host and device memory on accelerated systems.
In this post, we will:

1. Introduce a set of commonly used memory spaces
2. Identify what makes each memory space unique
3. Discuss some common use cases for each space

We primarily focus on AMD's MI200 line of GPUs, however many of the concepts discussed within this post carry over to other GPUs and APIs.

## Types of memory spaces

Working on heterogenous, accelerated systems implies that there are different memory and execution spaces available.
Special care must be taken when managing memory to ensure that data is in the right place at the right time.
While there are many different types of memory allocators and options in HIP, on AMD's MI200 they are all combinations of the following three properties:

1. Host vs device memory
   - Host memory exists on the host (e.g. CPU) of the machine, usually in random access memory (RAM).
   - Device memory exists on a device or accelerator (e.g. GPU) attached to the host. In the case of GPUs this memory is found in video random access memory (VRAM), which on recent GPU architectures is usually either:
      - Graphics double data rate (GDDR) synchronous dynamic random-access memory (SDRAM) - e.g. GDDR6 on AMD RDNA™ 2 GPUs
      - High-bandwidth memory (HBM) - e.g. HBM2e on AMD's MI200 GPUs
2. Pageable vs pinned (host) memory
   - Pageable memory is what we usually get when we call `malloc` or `new` in a C++ application. Pageable memory is unique in that it exists on "pages" (blocks of memory), which can be migrated to other memory storages. Examples of this are migrating memory between CPU sockets on a motherboard, or when a system runs out of space in RAM and starts dumping pages of RAM into the swap partition of your hard drive.
   - Pinned memory (or page-locked memory) is stored in pages that are locked to specific sectors in RAM and cannot be migrated.
3. Coarse-grained vs fine-grained coherence
   - Coarse-grained coherence means that memory is only considered up to date at kernel boundaries, which can be enforced through `hipDeviceSynchronize`, `hipStreamSynchronize`, or any blocking operation that acts on the null stream (e.g. `hipMemcpy`). For example, cacheable memory is a type of coarse-grained memory where an up-to-date copy of the data can be stored elsewhere (e.g. in an L2 cache).
   - Fine-grained coherence means the coherence is supported while a CPU/GPU kernel is running. This can be useful if both host and device are operating on the same dataspace using system-scope atomic operations (e.g. updating an error code or flag to a buffer). Fine-grained memory implies that up-to-date data may be made visible to others regardless of kernel boundaries as discussed above.

These memory properties are not mutually exclusive, which leads to a bit of complexity that we will attempt to clarify.

Before we look at how the HIP API works with these spaces, we need to introduce some important details about MI210, MI250, and MI250X GPUs.
The [MI210](https://www.amd.com/en/products/server-accelerators/amd-instinct-mi210) GPU is a standard PCIe-4.0 x16 card that wraps a single Graphics Compute Die (GCD) tied to 64GB of onboard HBM2e memory.
The [MI250](https://www.amd.com/en/products/server-accelerators/instinct-mi250) and [MI250X](https://www.amd.com/en/products/server-accelerators/instinct-mi250x) GPUs are [OCP Accelerator Modules (OAMs)](https://www.opencompute.org/documents/ocp-accelerator-module-design-specification-v1p5-final-20220223-docx-1-pdf) comprised of two GCDs with 128 GB of total memory but are presented to software as two unique devices with separate 64GB blocks of VRAM.
In this blog, we will use the term GPU to refer to the entire GPU and use GCD when the distinction between GPU and GCD is important.

In the following sections, we introduce allocators and de-allocators for using the various memory spaces available in HIP.

### Pageable memory

Pageable host memory in HIP uses the standard allocator and deallocator:

```c++
template<typename T>
T *
allocateHost_Pageable(const size_t size)
{
  return new T[size];
}

template<typename T>
void
deallocateHost_Pageable(T * ptr)
{
  delete [] ptr;
}
```

Note that we can adjust pageable memory alignment to improve performance when working with GPUs, however, we will hold that discussion for a future blog post.
By default, pageable memory is not accessible from device, but in the following sections, we will introduce [registering pageable memory](#registered-pageable-memory) and [enabling page migration](#enabling-page-migration) which workaround this restriction.

### Non-pageable (pinned) memory

Non-pageable memory (aka pinned memory or page-locked memory) is host memory that is mapped into the address space of all GPUs, meaning that the pointer can be used on both host and device.
Accessing host-resident pinned memory in device kernels is generally not recommended for performance, as it can force the data to traverse the host-device interconnect (e.g. PCIe), which is much slower than the on-device bandwidth (>40x on MI200).

Pinned host memory can be allocated with one of two types of coherence support:
1. `hipHostMallocCoherent`
   - Coherent pinned memory (aka zero-copy access memory) means that the host memory is not cached locally on the GPU, which implies fine-grained coherence.
   - Fine-grained coherence means the CPU can access up-to-date data in an allocation *while* a kernel is using the data on the GPU.
2. `hipHostMallocNonCoherent`
   - Non-coherent pinned memory means the GPU is free to locally store host data in MI200's L2 cache while it is in use.
   - The host may not see an up-to-date allocation while a kernel is running on device, and must wait until after the kernel completes or the cache is flushed (e.g. via a device or stream synchronize call).

Pinned memory allocations are coherent memory by default (`hipHostMallocDefault`).
Within HIP there are additional pinned memory flags (e.g. `hipHostMallocMapped` and `hipHostMallocPortable`), however for MI200 these options (on or off) do not impact performance, so we will ignore them.
The [HIP Programming Guide](https://github.com/ROCm-Developer-Tools/HIP/blob/main/docs/markdown/hip_programming_guide.md) has more information on pinned memory allocation flags.
Allocating coherent and non-coherent memory is controlled using the above flags with the `hipHostMalloc` call:
```c++
template<typename T>
T *
allocateHost_PinnedCoherent(const size_t size)
{
  void * ptr;
  HIP_CHECK_ERROR(hipHostMalloc(&ptr, size*sizeof(T), hipHostMallocCoherent));
  return reinterpret_cast<T*>(ptr);
}

template<typename T>
T *
allocateHost_PinnedNonCoherent(const size_t size)
{
  void * ptr;
  HIP_CHECK_ERROR(hipHostMalloc(&ptr, size*sizeof(T), hipHostMallocNonCoherent));
  return reinterpret_cast<T*>(ptr);
}

template<typename T>
void
deallocateHost_Pinned(T * ptr)
{
  HIP_CHECK_ERROR(hipHostFree((void*)ptr));
}
```

Much like how a process can be locked to a CPU core by setting affinity (e.g. through `taskset`), a pinned memory allocator does this with the memory storage system.
On multi-socket systems it is important to ensure that pinned memory is located on the same socket as the owning process, or else each cache line will be moved through the CPU-CPU interconnect, thereby increasing latency and potentially decreasing bandwidth.

In practice, pinned memory (coherent or non-coherent) is used to improve transfer times between host and device.
For transfer operations, such as `hipMemcpy` or `hipMemcpyAsync`, using pinned memory instead of pageable memory on host can lead to a ~3x improvement in bandwidth.

### Registered pageable memory

Registered pageable memory, as the name suggests, is a way of registering pageable memory with a GPU such that it can be directly accessed by a device kernel.
Registration ensures that the GPU is aware of the host pointer, which will *effectively* turn the pageable allocation into a pinned allocation.

To allocate registered memory, we must first allocate pageable memory, then register it with the currently active GPU.
```c++
template<typename T>
T *
allocateHost_Registered(size_t size,
                        const int device_id)
{
  T * ptr = allocateHost_Pageable<T>(size);
  HIP_CHECK_ERROR(hipSetDevice(device_id));
  HIP_CHECK_ERROR(hipHostRegister((void*)ptr, size*sizeof(T), hipHostRegisterDefault));
  return ptr;
}

template<typename T>
void
deallocateHost_Registered(T * ptr)
{
  HIP_CHECK_ERROR(hipHostUnregister((void*)ptr));
  delete [] ptr;
}
```

While this registration maps the host data to device, it does not necessarily mean that a kernel running on device can use the existing host pointer.
Instead, the registered device pointer can be retrieved given the host pointer:

```c++
template<typename T>
T *
getRegisteredDevicePtr(T * host_ptr)
{
  void * dev_ptr;
  HIP_CHECK_ERROR(hipHostGetDevicePointer(&dev_ptr, host_ptr, 0));
  return reinterpret_cast<T*>(dev_ptr);
}
```

The purpose of registering pageable memory is to ensure that the data can be accessed and modified from the GPU.
Registered memory is treated as `hipHostMallocCoherent` pinned memory, with equivalent performance.
The main reason for registering pageable memory is for situations where a developer is not in control of the allocator for a given allocation but still needs the memory to be accessible on device.

### Managed memory

Managed memory refers to universally addressable, or unified memory available on the MI200 series of GPUs.
Much like `hipHostMallocCoherent` pinned memory, managed memory shares a pointer between host and device and (by default) supports fine-grained coherence, however, managed memory can also automatically migrate pages between host and device.

Managed memory is not available on all systems, so it is advised to add a check for managed memory to your code:
```c++
bool
managedMemoryEnabled(const int device_id)
{
  int managed_memory = 0;
  HIP_CHECK_ERROR(hipDeviceGetAttribute(&managed_memory, hipDeviceAttributeManagedMemory, device_id));
  return managed_memory != 0;
}
```

Systems built using AMD's MI200 line of GPUs generally support managed memory, though with some caveats that we [discuss below](#enabling-page-migration).
Allocating managed memory uses `hipMallocManaged`:
```c++
template<typename T>
T *
allocateManaged(size_t size,
                const int device_id)
{
  if(!managedMemoryEnabled(device_id))
    throw std::logic_error("ERROR: Managed memory is not available on this device.");

  HIP_CHECK_ERROR(hipSetDevice(device_id));
  void * ptr;
  HIP_CHECK_ERROR(hipMallocManaged((void**)&ptr, size * sizeof(T)));
  return reinterpret_cast<T*>(ptr);
}

template<typename T>
void
deallocateManaged(T * ptr)
{
  HIP_CHECK_ERROR(hipFree((void*)ptr));
}
```

HIP supports additional calls that work with page migration in HIP to enable things like prioritizing memory locations (`hipMemAdvise`), prefetching data to device/host (`hipMemPrefetchAsync`), and acquiring information about memory location (`hipMemRangeGetAttribute`). 
A more detailed study of managed memory and page migration will be done in a future blog.
In the meantime, please see the [additional resources](#additional-resources) section.

Managed memory is used in cases where we want the HIP to automatically transfer ownership of data between host and device on demand, thereby simplifying memory management for the user.
This memory space greatly simplifies the porting process when transitioning from CPU to GPU workloads.

### Device memory

Device memory is simply memory that is allocated on a specific device.
Much like pinned host memory, device memory can be allocated as fine-grained or coarse-grained.
For performance reasons, we generally don't want to restrict the cacheability of data on device, so the device allocator `hipMalloc` returns coarse-grained memory:
```c++
template<typename T>
T *
allocateDevice(const size_t size,
               const int device_id)
{
  HIP_CHECK_ERROR(hipSetDevice(device_id));
  void * ptr;
  HIP_CHECK_ERROR(hipMalloc(&ptr, size*sizeof(T)));
  return reinterpret_cast<T*>(ptr);
}

template<typename T>
void
deallocateDevice(T * ptr)
{
  HIP_CHECK_ERROR(hipFree((void*)ptr));
}
```

Alternatively, we can allocate fine-grained memory on supported systems using the extended malloc call `hipExtMallocWithFlags` with the `hipDeviceMallocFinegrained` flag.
Support for coarse-grained and fine-grained memory on CPUs and GPUs can be found in the "Pool Info" section of `rocminfo`.
In the following example we see that the CPU has both coarse-grained and fine-grained memory pools available, while the GPU is limited to coarse-grained memory:
```bash
$ rocminfo
...
*******
Agent 1
*******
  Name:                    AMD EPYC 7742 64-Core Processor
...
  Pool Info:
    Pool 1
      Segment:                 GLOBAL; FLAGS: FINE GRAINED
...
    Pool 3
      Segment:                 GLOBAL; FLAGS: COARSE GRAINED
...
*******
Agent 9
*******
  Name:                    gfx90a
...
  Pool Info:
    Pool 1
      Segment:                 GLOBAL; FLAGS: COARSE GRAINED
...
```
By default `hipMalloc` and `hipFree` are blocking calls, however, HIP recently added non-blocking versions `hipMallocAsync` and `hipFreeAsync` which take in a stream as an additional argument.

Device memory should be used whenever possible.
Not only is it much more performant than accessing host memory on device, but it also gives more control over where memory is located in a system.

## Improving transfer bandwidth

In most cases, the default behavior for HIP in transferring data from a pinned host allocation to device will run at the limit of the interconnect.
However, there are certain cases where the interconnect is not the bottleneck.
To understand this, we will discuss how a GPU transfers memory to and from a host allocation.

The primary way to transfer data onto and off of a MI200 is to use the onboard System Direct Memory Access (SDMA) engine, which is used to feed blocks of memory to the off-device interconnect (either GPU-CPU or GPU-GPU).
Each MI200 GCD has a separate SDMA engine for host-to-device and device-to-host memory transfers.
Importantly, SDMA engines are separate from the computing infrastructure, meaning that memory transfers to/from a device will not impact kernel compute performance, though they do impact memory bandwidth to a limited extent.
The SDMA engines are mainly tuned for PCIe-4.0 x16, which means they are designed to operate at bandwidths up to 32 GB/s.

An important feature of the MI250X platform used in ORNL's Frontier supercomputer is the Infinity Fabric™ interconnect between host and device.
The Infinity Fabric interconnect supports improved performance over standard PCIe-4.0 (usually ~50% more bandwidth); however, since the SDMA engine does not run at this speed, it will not max out the bandwidth of the faster interconnect.

We can counter this bandwidth limitation by bypassing the SDMA engine and replacing it with a type of copy kernel known as a "blit" kernel.
Blit kernels will use the compute units on the GPU, thereby consuming compute resources, which may not always be beneficial.
The easiest way to enable blit kernels is to set an environment variable `HSA_ENABLE_SDMA=0`, which will disable the SDMA engine.
On systems where the GPU uses a PCIe interconnect instead of an Infinity Fabric interconnect, blit kernels will not impact bandwidth, but will still consume compute resources.
The use of SDMA vs blit kernels also applies to MPI data transfers and GPU-GPU transfers, but we will save this discussion for a future blog post.

## Enabling page migration

On MI200 GPUs there is an option to automatically migrate pages of memory between host and device.
This is important for managed memory, where the locality of the data is important for performance.
Depending on the system, page migration may be disabled by default in which case managed memory will act like pinned host memory and suffer degraded performance.

Enabling page migration allows the GPU (or host) to retry after a page fault (normally a memory access error), and instead retrieve the missing page.
On MI200 platforms we can enable page migration by setting the environment variable `HSA_XNACK=1`.
While this environment variable is required at kernel runtime to enable page migration, it is also helpful to enable this environment variable at compile time, which can change the performance of any compiled kernels.

To check if page migration is available on a MI200 platform, we can use `rocminfo` in a Linux terminal:
```bash
$ rocminfo | grep xnack
      Name:                    amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack-
```
Here, `xnack-` means that XNACK is available but is disabled by default. 
Enabling XNACK gives the expected result:
```bash
$ HSA_XNACK=1 rocminfo | grep xnack
      Name:                    amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack+
```

It is also important to note that enabling page migration also affects pageable host memory, meaning that it will automatically be migrated to the GPU on demand.
A side effect of this is that if you disable XNACK, and attempt to use pageable memory on device, you will have undefined behavior (e.g. segfaults and invalid pointer error codes).
Page migration is not always available - e.g. on the AMD RDNA™ 2 GPUs or in operating systems that do not support [heterogeneous memory management (HMM)](https://www.kernel.org/doc/html/v5.0/vm/hmm.html).

## Summary

We have explored a common set of memory spaces used with the HIP API on AMD's MI200 platforms.
We introduced how each memory space is allocated and deallocated and discussed what each space is designed to do.
We also discussed how SDMA engines can limit bandwidth on certain MI250X platforms, and how enabling page migration can significantly improve performance with managed memory.
Here is a summary of simple do's and don't's for working with various memory spaces on MI200 systems:

Some do's:
1. If the application needs to move data back and forth between device and host (separate allocations), use pinned memory on the host side.
2. If the application needs to use data on both host and device regularly, doesn't want to deal with separate allocations, and is not worried about maxing out the VRAM on MI200 GPUs (64 GB per GCD), use managed memory.
3. If using a MI250X system (e.g. ORNL's Frontier supercomputer), check to see if turning off SDMA improves performance for host-device and MPI data transfers.
4. If managed memory performance is poor, check to see if managed memory is supported on your system and if page migration (XNACK) is enabled.

Some don'ts:
1. If you want to make use of page migration on MI200, use managed memory. While pageable memory will migrate correctly, it is not a portable solution and can have performance issues if it isn't page aligned.
2. Try to design your algorithms to avoid host-device memory coherence (e.g. system scope atomics). While it can be a useful feature in very specific cases, it is not supported on all systems, and can negatively impact performance by introducing the host-device interconnect bottleneck.

This blog is a very high-level overview of memory spaces on MI200, and we plan to do a deeper dive into managed memory, atomic operations, memory coherence, and performance in later posts.

If you have any questions or comments, please reach out to us on GitHub [Discussions](https://github.com/amd/amd-lab-notes/discussions)

## Additional resources

- [HIP Programming Guide](https://github.com/ROCm-Developer-Tools/HIP/blob/main/docs/markdown/hip_programming_guide.md)
- [ENCCS AMD Node Memory Model](https://enccs.github.io/AMD-ROCm-development/memory_model/)
- [Crusher Quick Start Guide](https://docs.olcf.ornl.gov/systems/crusher_quick_start_guide.html)
- [Heterogeneous Memory Management (HMM)](https://www.kernel.org/doc/html/v5.0/vm/hmm.html)

*AMD, AMD Instinct, RDNA, Infinity Fabric, and combinations thereof are trademarks of Advanced Micro Devices, Inc.*

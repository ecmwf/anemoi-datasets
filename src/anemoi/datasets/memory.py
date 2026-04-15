# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""TODO: move that code to anemoi-utils"""

import os
import resource

import psutil


def available_memory():
    """Returns the memory available to the current process in bytes.
    Accounts for Slurm, Cgroups, and System limits.
    """
    limits = []

    # 1. Check Slurm Limits
    # Slurm sets variables in MB. We convert to bytes.
    slurm_mem_per_node = os.environ.get("SLURM_MEM_PER_NODE")
    if slurm_mem_per_node:
        limits.append(int(slurm_mem_per_node) * 1024 * 1024)

    slurm_mem_per_cpu = os.environ.get("SLURM_MEM_PER_CPU")
    if slurm_mem_per_cpu:
        cpus = int(os.environ.get("SLURM_CPUS_ON_NODE", 1))
        limits.append(int(slurm_mem_per_cpu) * cpus * 1024 * 1024)

    # 2. Check Cgroup Limits (Common on Linux/Docker/Slurm)
    # Cgroup v2 is standard on modern kernels.
    cgroup_limit_path = "/sys/fs/cgroup/memory.max"  # Cgroup v2
    if os.path.exists(cgroup_limit_path):
        with open(cgroup_limit_path) as f:
            val = f.read().strip()
            if val != "max":
                limits.append(int(val))
    else:
        # Fallback to Cgroup v1
        cgroup_v1_path = "/sys/fs/cgroup/memory/memory.limit_in_bytes"
        if os.path.exists(cgroup_v1_path):
            with open(cgroup_v1_path) as f:
                limits.append(int(f.read().strip()))

    # 3. Check RLIMIT_AS (ulimit -v)
    # This is the address space limit. soft_limit is what we care about.
    soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_AS)
    if soft_limit != resource.RLIM_INFINITY:
        limits.append(soft_limit)

    # 4. Physical System Memory (The ultimate ceiling)
    # We use psutil.virtual_memory().available for what's actually free right now
    system_available = psutil.virtual_memory().available
    limits.append(system_available)

    # The actual limit is the smallest of all enforced constraints
    return min(limits)


if __name__ == "__main__":
    mem_bytes = available_memory()
    print(f"Available Memory: {mem_bytes / (1024**3):.2f} GB")

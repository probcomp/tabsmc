#!/usr/bin/env python
"""Diagnose GPU memory usage patterns with JAX and SMC."""

import jax
import jax.numpy as jnp
import numpy as np
import subprocess
import gc
import time


def get_gpu_memory_info():
    """Get current GPU memory usage."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            used, total = map(int, result.stdout.strip().split(','))
            return used, total
        return None, None
    except:
        return None, None


def print_memory_status(label):
    """Print current memory status."""
    used, total = get_gpu_memory_info()
    if used is not None:
        print(f"{label}: {used}/{total} MB ({used/total*100:.1f}%)")
    else:
        print(f"{label}: GPU memory info not available")


def test_array_allocation():
    """Test basic array allocation patterns."""
    print("\n=== Testing Basic Array Allocation ===")
    
    # Clear everything first
    gc.collect()
    jax.clear_caches()
    print_memory_status("Initial")
    
    # Test 1: Allocate arrays of different sizes
    sizes = [100, 1000, 10000]
    for size in sizes:
        print(f"\nAllocating array of size {size} x 1000 x 128...")
        arr = jnp.zeros((size, 1000, 128))
        arr.block_until_ready()
        print_memory_status(f"After allocating {size} x 1000 x 128")
        
        # Calculate expected memory
        expected_mb = (size * 1000 * 128 * 4) / (1024 * 1024)  # 4 bytes per float32
        print(f"Expected memory usage: ~{expected_mb:.1f} MB")
        
        del arr
        gc.collect()
        jax.clear_caches()
        time.sleep(1)  # Give time for memory to be freed
        print_memory_status("After cleanup")


def test_vmap_memory():
    """Test memory usage with vmap."""
    print("\n=== Testing vmap Memory Usage ===")
    
    gc.collect()
    jax.clear_caches()
    print_memory_status("Initial")
    
    def simple_function(x):
        return x @ x.T
    
    # Test with different numbers of "particles"
    for n_particles in [10, 100, 1000]:
        print(f"\nTesting vmap with {n_particles} elements...")
        
        # Create input
        x = jnp.ones((n_particles, 100, 100))
        print_memory_status(f"After creating input ({n_particles} x 100 x 100)")
        
        # Run vmap
        result = jax.vmap(simple_function)(x)
        result.block_until_ready()
        print_memory_status(f"After vmap computation")
        
        del x, result
        gc.collect()
        jax.clear_caches()
        time.sleep(1)
        print_memory_status("After cleanup")


def test_jit_compilation():
    """Test if JIT compilation is causing memory usage."""
    print("\n=== Testing JIT Compilation Effects ===")
    
    gc.collect()
    jax.clear_caches()
    print_memory_status("Initial")
    
    @jax.jit
    def allocate_and_compute(n):
        # This should fail if n is traced
        arr = jnp.zeros((n, 1000, 128))
        return jnp.sum(arr)
    
    # Test if function gets recompiled for different sizes
    for n in [10, 100, 1000]:
        print(f"\nTesting with n={n}...")
        try:
            result = allocate_and_compute(n)
            result.block_until_ready()
            print_memory_status(f"After computation with n={n}")
            print(f"Result: {result}")
        except Exception as e:
            print(f"Error: {e}")
        
        gc.collect()
        jax.clear_caches()


def test_static_vs_dynamic():
    """Test static vs dynamic array shapes."""
    print("\n=== Testing Static vs Dynamic Shapes ===")
    
    gc.collect()
    jax.clear_caches()
    print_memory_status("Initial")
    
    # Static shape function
    @jax.jit
    def static_allocate():
        return jnp.zeros((1000, 1000, 10))
    
    # Dynamic shape function (will fail with JIT)
    def dynamic_allocate(n):
        return jnp.zeros((n, 1000, 10))
    
    print("\nStatic allocation:")
    arr1 = static_allocate()
    arr1.block_until_ready()
    print_memory_status("After static allocation")
    
    print("\nDynamic allocation (100):")
    arr2 = dynamic_allocate(100)
    arr2.block_until_ready()
    print_memory_status("After dynamic allocation (100)")
    
    print("\nDynamic allocation (1000):")
    arr3 = dynamic_allocate(1000)
    arr3.block_until_ready()
    print_memory_status("After dynamic allocation (1000)")
    
    # Check actual memory usage
    print(f"\nArray shapes: {arr1.shape}, {arr2.shape}, {arr3.shape}")
    
    del arr1, arr2, arr3
    gc.collect()
    jax.clear_caches()


def check_jax_config():
    """Check JAX configuration that might affect memory."""
    print("\n=== JAX Configuration ===")
    print(f"JAX devices: {jax.devices()}")
    print(f"Default backend: {jax.default_backend()}")
    
    # Check if we're using GPU
    try:
        from jax.lib import xla_bridge
        print(f"XLA backend: {xla_bridge.get_backend().platform}")
    except:
        pass
    
    # Check memory allocation settings
    import os
    relevant_env_vars = [
        'XLA_PYTHON_CLIENT_PREALLOCATE',
        'XLA_PYTHON_CLIENT_MEM_FRACTION',
        'XLA_PYTHON_CLIENT_ALLOCATOR',
        'TF_FORCE_GPU_ALLOW_GROWTH',
        'CUDA_VISIBLE_DEVICES'
    ]
    
    print("\nEnvironment variables:")
    for var in relevant_env_vars:
        value = os.environ.get(var, 'Not set')
        print(f"  {var}: {value}")


def main():
    print("=== GPU Memory Diagnostic Tool ===")
    
    # Check configuration first
    check_jax_config()
    
    # Run tests
    test_array_allocation()
    test_vmap_memory()
    test_jit_compilation()
    test_static_vs_dynamic()
    
    print("\n=== Summary ===")
    print("If GPU memory doesn't change with array size, possible causes:")
    print("1. JAX is preallocating GPU memory (XLA_PYTHON_CLIENT_PREALLOCATE=false to disable)")
    print("2. Arrays are being allocated on CPU instead of GPU")
    print("3. JIT compilation is creating fixed-size allocations")
    print("4. Memory fragmentation or caching issues")
    
    # Final memory status
    print()
    print_memory_status("Final GPU memory")


if __name__ == "__main__":
    main()
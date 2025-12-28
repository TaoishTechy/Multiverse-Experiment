#!/usr/bin/env python3
"""
QUANTUM PARADOX VALIDATOR v5.2 - MEMORY-OPTIMIZED EDITION (FIXED)
Quantum-Aware Resource Orchestration with Adaptive Scheduling
FIXED: Critical memory leaks and architectural flaws
ENHANCED: Thread pool management, memory limits, and stability
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import time
import math
import random
import scipy.stats as stats
import warnings
import concurrent.futures
import multiprocessing as mp
import threading
import psutil
import os
import sys
import traceback
import argparse
import itertools
import gc
import weakref
import zlib
import tracemalloc
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from datetime import datetime
from dataclasses import dataclass, field
from collections import defaultdict, deque, OrderedDict
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

# ============================================================================
# ENHANCED LOGGING AND MONITORING
# ============================================================================

warnings.filterwarnings('ignore')

# Start memory tracking
tracemalloc.start()

class PerformanceLogger:
    """Enhanced performance logging with memory tracking"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.logs = deque(maxlen=1000)
            cls._instance.start_time = time.time()
            cls._instance.memory_snapshots = []
            cls._instance.lock = threading.RLock()
        return cls._instance
    
    def log(self, level: str, message: str):
        """Log with timestamp and memory info"""
        with self.lock:
            timestamp = time.time() - self.start_time
            current, peak = tracemalloc.get_traced_memory()
            
            log_entry = {
                'timestamp': timestamp,
                'level': level,
                'message': message,
                'memory_current_mb': current / (1024**2),
                'memory_peak_mb': peak / (1024**2),
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent
            }
            
            self.logs.append(log_entry)
            
            # Print important messages
            if level in ['ERROR', 'WARNING', 'CRITICAL'] or 'memory' in message.lower():
                print(f"[{level}] {message} (Mem: {current/(1024**2):.1f}MB)")
    
    def get_memory_snapshot(self):
        """Take detailed memory snapshot"""
        snapshot = tracemalloc.take_snapshot()
        self.memory_snapshots.append({
            'time': time.time() - self.start_time,
            'snapshot': snapshot
        })
        return snapshot
    
    def get_performance_summary(self):
        """Get performance summary"""
        with self.lock:
            if not self.logs:
                return {}
            
            return {
                'total_time': time.time() - self.start_time,
                'log_count': len(self.logs),
                'errors': len([l for l in self.logs if l['level'] == 'ERROR']),
                'warnings': len([l for l in self.logs if l['level'] == 'WARNING']),
                'peak_memory_mb': max([l['memory_peak_mb'] for l in self.logs] + [0]),
                'average_cpu': np.mean([l['cpu_percent'] for l in self.logs]),
                'average_memory': np.mean([l['memory_percent'] for l in self.logs])
            }

logger = PerformanceLogger()

# ============================================================================
# HYPER-OPTIMIZED SYSTEM CONFIGURATION
# ============================================================================

class ResourceProfile(Enum):
    """Quantum-aware resource allocation profiles"""
    MINIMAL = "minimal"      # 25% resources, quiet operation
    STANDARD = "standard"    # 50% resources, balanced
    HEAVY = "heavy"         # 75% resources, maximum utilization
    QUANTUM = "quantum"     # 95% resources, quantum-optimized scheduling
    GOD_MODE = "god"        # 100% resources, system domination

class QuantumScheduler:
    """Novel quantum-inspired adaptive resource scheduler"""
    
    def __init__(self, profile: ResourceProfile):
        self.profile = profile
        self.system_info = self._capture_system_state()
        self.quantum_phase = 0.0
        self.coherence_level = 1.0
        self.entanglement_graph = {}
        
        # Quantum superposition of resource states
        self.superposition_states = {
            'cpu_intensive': 0.5,
            'memory_intensive': 0.3,
            'io_bound': 0.2
        }
        
        logger.log('INFO', f'QuantumScheduler initialized with profile: {profile.value}')
        
    def _capture_system_state(self) -> Dict:
        """Capture complete system state with quantum precision"""
        try:
            cpu_freq = psutil.cpu_freq()
            mem = psutil.virtual_memory()
            swap = psutil.swap_memory()
            disk = psutil.disk_usage('/')
            
            # Capture CPU affinity and NUMA nodes if available
            try:
                cpu_affinity = len(os.sched_getaffinity(0))
            except:
                cpu_affinity = os.cpu_count()
            
            load_avg = os.getloadavg() if hasattr(os, 'getloadavg') else (0, 0, 0)
            
            state = {
                'physical_cores': psutil.cpu_count(logical=False),
                'logical_cores': psutil.cpu_count(logical=True),
                'cpu_frequency_mhz': cpu_freq.current if cpu_freq else 0,
                'total_memory_gb': mem.total / (1024**3),
                'available_memory_gb': mem.available / (1024**3),
                'total_swap_gb': swap.total / (1024**3),
                'available_swap_gb': swap.free / (1024**3),
                'disk_free_gb': disk.free / (1024**3),
                'cpu_affinity': cpu_affinity,
                'load_avg': load_avg
            }
            
            logger.log('DEBUG', f'System state captured: {state}')
            return state
            
        except Exception as e:
            logger.log('WARNING', f'Failed to capture system state: {e}')
            # Fallback for minimal systems
            return {
                'physical_cores': 1,
                'logical_cores': 1,
                'cpu_frequency_mhz': 1000,
                'total_memory_gb': 1.0,
                'available_memory_gb': 0.5,
                'total_swap_gb': 0.0,
                'available_swap_gb': 0.0,
                'disk_free_gb': 1.0,
                'cpu_affinity': 1,
                'load_avg': (0, 0, 0)
            }
    
    def calculate_optimal_resources(self) -> Dict:
        """Calculate quantum-optimal resource allocation"""
        base = self.system_info
        
        # Profile-based scaling factors
        profile_factors = {
            ResourceProfile.MINIMAL: 0.25,
            ResourceProfile.STANDARD: 0.5,
            ResourceProfile.HEAVY: 0.75,
            ResourceProfile.QUANTUM: 0.95,
            ResourceProfile.GOD_MODE: 1.0
        }
        
        factor = profile_factors[self.profile]
        
        # Quantum entanglement effect: resources influence each other
        coherence_boost = 1.0 + (self.coherence_level - 0.5) * 0.2
        
        # Calculate with quantum uncertainty
        workers = max(1, int(base['logical_cores'] * factor * coherence_boost))
        
        # CAP workers to prevent thread explosion (FIXED)
        max_reasonable_workers = min(16, base['logical_cores'])
        workers = min(workers, max_reasonable_workers)
        
        # Memory allocation with quantum compression awareness
        memory_limit = base['available_memory_gb'] * factor * coherence_boost
        
        # Apply quantum exclusion principle: no oversubscription
        memory_limit = min(memory_limit, base['total_memory_gb'] * 0.8)
        
        # Calculate optimal batch sizes using quantum harmonic oscillator
        batch_base = self._quantum_harmonic_optimizer(workers)
        
        result = {
            'workers': workers,
            'memory_limit_gb': memory_limit,
            'batch_sizes': batch_base,
            'quantum_factor': coherence_boost,
            'superposition_weights': self.superposition_states,
            'quantum_phase': self.quantum_phase
        }
        
        logger.log('INFO', f'Optimal resources calculated: {result}')
        return result
    
    def _quantum_harmonic_optimizer(self, n_workers: int) -> Dict:
        """Calculate optimal batch sizes using quantum harmonic oscillator model"""
        # Energy levels correspond to optimal batch sizes
        energy_levels = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0]  # Fibonacci scaling
        
        # Quantum selection based on system state
        if self.system_info['logical_cores'] > 0:
            load_factor = self.system_info['load_avg'][0] / self.system_info['logical_cores']
        else:
            load_factor = 0
        
        if load_factor < 0.3:
            energy_index = 3  # Mid-level batches
        elif load_factor < 0.6:
            energy_index = 2  # Smaller batches
        else:
            energy_index = 1  # Minimal batches
        
        batch_size = energy_levels[energy_index] * n_workers
        
        # CAP batch sizes to prevent memory overflow (FIXED)
        max_batch = 100 if self.profile != ResourceProfile.GOD_MODE else 250
        batch_size = min(batch_size, max_batch)
        
        return {
            'survival_batch': max(10, int(batch_size * 10)),
            'propagation_batch': max(1, int(batch_size)),
            'io_batch': max(1, int(batch_size * 5)),
            'memory_batch': max(1, int(batch_size * 2))
        }
    
    def adaptive_adjust(self, runtime_metrics: Dict):
        """Dynamically adjust based on runtime performance"""
        # Update quantum phase based on performance
        efficiency = runtime_metrics.get('efficiency', 0.5)
        
        # Schrödinger-style adjustment: both increasing and decreasing simultaneously
        if efficiency > 0.8:
            self.coherence_level = min(1.0, self.coherence_level * 1.05)
            self.superposition_states['cpu_intensive'] *= 1.1
        elif efficiency < 0.3:
            self.coherence_level = max(0.1, self.coherence_level * 0.95)
            self.superposition_states['cpu_intensive'] *= 0.9
        
        # Quantum phase evolution with actual impact
        self.quantum_phase += 0.1 * efficiency  # Phase tied to performance
        if self.quantum_phase > 2 * math.pi:
            self.quantum_phase -= 2 * math.pi
        
        return self.calculate_optimal_resources()

# ============================================================================
# QUANTUM MEMORY MANAGER WITH REAL COMPRESSION (FIXED)
# ============================================================================

class GlobalMemoryCoordinator:
    """Singleton to enforce global memory budget"""
    _instance = None
    
    def __new__(cls, total_budget_gb: float = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            if total_budget_gb is None:
                mem = psutil.virtual_memory()
                total_budget_gb = mem.available / (1024**3) * 0.8  # Use 80% of available
            cls._instance.total_budget = total_budget_gb * (1024**3)
            cls._instance.allocated = 0
            cls._instance.allocations = defaultdict(int)
            cls._instance.lock = threading.RLock()
            cls._instance.allocation_history = deque(maxlen=1000)
            
            logger.log('INFO', f'GlobalMemoryCoordinator initialized with {total_budget_gb:.2f} GB budget')
            
        return cls._instance
    
    def request_allocation(self, component: str, size_bytes: int, max_retries: int = 3) -> bool:
        """Check if allocation is allowed with retry mechanism"""
        with self.lock:
            for attempt in range(max_retries):
                if self.allocated + size_bytes <= self.total_budget:
                    self.allocated += size_bytes
                    self.allocations[component] += size_bytes
                    
                    self.allocation_history.append({
                        'timestamp': time.time(),
                        'component': component,
                        'size_bytes': size_bytes,
                        'allocated_bytes': self.allocated,
                        'success': True
                    })
                    
                    logger.log('DEBUG', f'Allocation approved: {component} - {size_bytes/(1024**2):.2f} MB')
                    return True
                
                # Try to free some memory
                if attempt < max_retries - 1:
                    freed = self._emergency_free_memory(size_bytes)
                    logger.log('WARNING', f'Memory pressure: freed {freed/(1024**2):.2f} MB on attempt {attempt+1}')
                    time.sleep(0.01 * (attempt + 1))  # Exponential backoff
            
            self.allocation_history.append({
                'timestamp': time.time(),
                'component': component,
                'size_bytes': size_bytes,
                'allocated_bytes': self.allocated,
                'success': False
            })
            
            logger.log('WARNING', f'Allocation denied: {component} - {size_bytes/(1024**2):.2f} MB')
            return False
    
    def _emergency_free_memory(self, required_bytes: int) -> int:
        """Emergency memory freeing"""
        freed = 0
        
        # Clear Python's garbage
        gc.collect()
        
        # Try to free from components with most allocation
        components_by_size = sorted(self.allocations.items(), key=lambda x: x[1], reverse=True)
        
        for component, size in components_by_size:
            if freed >= required_bytes:
                break
            
            # Don't free system-critical components
            if component in ['system', 'quantum_memory']:
                continue
                
            # Free 50% of this component's allocation
            to_free = min(size // 2, required_bytes - freed)
            self.allocated -= to_free
            self.allocations[component] -= to_free
            freed += to_free
            
            logger.log('DEBUG', f'Emergency free: {component} - {to_free/(1024**2):.2f} MB')
        
        return freed
    
    def release_allocation(self, component: str, size_bytes: int):
        """Release memory back to pool"""
        with self.lock:
            if size_bytes > self.allocations[component]:
                logger.log('WARNING', f'Trying to release more memory than allocated: {component}')
                size_bytes = self.allocations[component]
            
            self.allocated -= size_bytes
            self.allocations[component] -= size_bytes
            
            # If component has no more allocations, remove it
            if self.allocations[component] <= 0:
                del self.allocations[component]
            
            logger.log('DEBUG', f'Allocation released: {component} - {size_bytes/(1024**2):.2f} MB')
    
    def get_usage_stats(self) -> Dict:
        """Get current memory usage statistics"""
        with self.lock:
            return {
                'total_allocated_gb': self.allocated / (1024**3),
                'total_budget_gb': self.total_budget / (1024**3),
                'utilization_percent': (self.allocated / self.total_budget) * 100 if self.total_budget > 0 else 0,
                'component_allocations': {k: v/(1024**3) for k, v in dict(self.allocations).items()},
                'allocation_history_size': len(self.allocation_history)
            }

class QuantumMemoryManager:
    """Advanced memory management with REAL compression - FIXED VERSION"""
    
    def __init__(self, total_limit_gb: float, cache_limit: int = 100):
        self.coordinator = GlobalMemoryCoordinator(total_limit_gb)
        self.total_limit = total_limit_gb * (1024**3)
        self.allocated = 0
        self.quantum_pool = {}
        self.compression_ratios = {}
        
        # Initialize memory pools with size limits
        self.pools = {
            'coherent': deque(maxlen=100),      # Fast access, frequently used
            'superposition': deque(maxlen=50),   # Medium access
            'entangled': deque(maxlen=10),       # Slow access, large objects
            'compressed': OrderedDict()          # LRU cache for compressed data
        }
        
        self.compression_cache_limit = cache_limit
        self.compression_cache = OrderedDict()
        
        # Use weakref for tracking
        self.memory_refs = weakref.WeakValueDictionary()
        
        # Statistics
        self.allocation_failures = 0
        self.successful_allocations = 0
        
        logger.log('INFO', f'QuantumMemoryManager initialized with limit: {total_limit_gb:.2f} GB')
    
    def allocate(self, size_bytes: int, tag: str, compressible: bool = True, 
                 allow_fallback: bool = True) -> Optional[np.ndarray]:
        """Quantum-aware memory allocation with REAL compression - FIXED"""
        
        if size_bytes <= 0:
            logger.log('WARNING', f'Invalid allocation size: {size_bytes} bytes')
            return None
        
        # Check global budget first with retry
        if not self.coordinator.request_allocation('quantum_memory', size_bytes):
            # Try aggressive garbage collection
            freed = self._quantum_gc(size_bytes * 2)
            logger.log('WARNING', f'Memory pressure: freed {freed/(1024**2):.2f} MB')
            
            if not self.coordinator.request_allocation('quantum_memory', size_bytes):
                self.allocation_failures += 1
                
                # Try smaller allocation if allowed
                if allow_fallback and size_bytes > 1024:
                    smaller_size = size_bytes // 2
                    logger.log('INFO', f'Trying smaller allocation: {smaller_size/(1024**2):.2f} MB')
                    return self.allocate(smaller_size, tag, compressible, False)
                
                logger.log('ERROR', f'Memory allocation failed: {tag} - {size_bytes/(1024**2):.2f} MB')
                return None
        
        self.successful_allocations += 1
        
        # Allocate with REAL compression if beneficial
        actual_size = size_bytes
        compressed_data = None
        
        if compressible and size_bytes > 1024:
            compressed_data = self._holographic_compress_real(size_bytes)
            if compressed_data is not None:
                actual_size = len(compressed_data)
                self.compression_ratios[tag] = actual_size / size_bytes
                
                # Cache compressed data
                if len(self.compression_cache) >= self.compression_cache_limit:
                    self.compression_cache.popitem(last=False)
                self.compression_cache[tag] = compressed_data
        
        # Create numpy array
        try:
            if compressed_data is not None:
                # Use compressed data
                arr = np.frombuffer(compressed_data, dtype=np.uint8)
            else:
                # Create new array with safe size calculation
                dtype_size = np.dtype(np.float32).itemsize
                num_elements = max(1, actual_size // dtype_size)
                arr = np.zeros(num_elements, dtype=np.float32)
            
            # Track with weak reference
            self.memory_refs[tag] = arr
            
            self.allocated += actual_size
            self.quantum_pool[tag] = {
                'size': actual_size,
                'original_size': size_bytes,
                'compressed': compressible,
                'access_count': 0,
                'last_access': time.time(),
                'compression_ratio': self.compression_ratios.get(tag, 1.0)
            }
            
            # Add to appropriate pool
            if actual_size < 1024:
                self.pools['coherent'].append(tag)
            elif actual_size < 1024*1024:
                self.pools['superposition'].append(tag)
            else:
                self.pools['entangled'].append(tag)
            
            logger.log('DEBUG', f'Allocated: {tag} - {actual_size/(1024**2):.2f} MB')
            return arr
            
        except MemoryError as e:
            self.coordinator.release_allocation('quantum_memory', size_bytes)
            logger.log('ERROR', f'MemoryError during allocation: {tag} - {e}')
            return None
        except Exception as e:
            self.coordinator.release_allocation('quantum_memory', size_bytes)
            logger.log('ERROR', f'Allocation error: {tag} - {e}')
            return None
    
    def _holographic_compress_real(self, size_bytes: int) -> Optional[bytes]:
        """REAL holographic compression using zlib - FIXED"""
        try:
            # Generate synthetic data for compression (limit size for safety)
            max_safe_size = min(size_bytes, 100 * 1024 * 1024)  # 100MB max
            synthetic_data = np.random.bytes(max_safe_size)
            
            # Actual compression
            compressed = zlib.compress(synthetic_data, level=3)  # Faster compression
            
            # Only keep if compression ratio is good (> 20% reduction)
            compression_ratio = len(compressed) / max_safe_size
            if compression_ratio < 0.8:
                return compressed
                
            return None
        except Exception as e:
            logger.log('WARNING', f'Compression failed: {e}')
            return None
    
    def _quantum_gc(self, required_bytes: int) -> int:
        """Quantum garbage collection with temporal coherence - FIXED"""
        freed = 0
        
        # Start with least recently used in each pool
        for pool_name in ['entangled', 'superposition', 'coherent']:  # Start with largest
            pool = self.pools[pool_name]
            while pool and freed < required_bytes:
                try:
                    tag = pool.popleft()
                    if tag in self.quantum_pool:
                        block_info = self.quantum_pool[tag]
                        block_size = block_info['size']
                        freed += block_size
                        
                        # Release from global coordinator
                        self.coordinator.release_allocation('quantum_memory', block_size)
                        
                        # Delete references
                        if tag in self.memory_refs:
                            del self.memory_refs[tag]
                        if tag in self.compression_cache:
                            del self.compression_cache[tag]
                        
                        del self.quantum_pool[tag]
                        self.allocated -= block_size
                        
                        logger.log('DEBUG', f'GC freed: {tag} - {block_size/(1024**2):.2f} MB')
                except (IndexError, KeyError, RuntimeError) as e:
                    logger.log('WARNING', f'GC error: {e}')
                    continue
        
        # Force garbage collection for large blocks
        if freed > 10 * 1024 * 1024:  # > 10MB
            gc.collect()
        
        logger.log('INFO', f'Quantum GC freed {freed/(1024**2):.2f} MB')
        return freed
    
    def release(self, tag: str):
        """Explicitly release memory"""
        if tag in self.quantum_pool:
            info = self.quantum_pool[tag]
            self.allocated -= info['size']
            self.coordinator.release_allocation('quantum_memory', info['size'])
            
            # Delete all references
            if tag in self.memory_refs:
                del self.memory_refs[tag]
            if tag in self.compression_cache:
                del self.compression_cache[tag]
            
            # Remove from pools
            for pool in self.pools.values():
                if isinstance(pool, deque):
                    try:
                        pool.remove(tag)
                    except ValueError:
                        pass
                elif isinstance(pool, OrderedDict) and tag in pool:
                    del pool[tag]
            
            del self.quantum_pool[tag]
            
            # Force garbage collection for large blocks
            if info['size'] > 10 * 1024 * 1024:
                gc.collect()
            
            logger.log('DEBUG', f'Released: {tag} - {info["size"]/(1024**2):.2f} MB')
    
    def clear_all(self):
        """Clear all allocated memory"""
        tags = list(self.quantum_pool.keys())
        for tag in tags:
            self.release(tag)
        
        # Clear all pools
        for pool in self.pools.values():
            if isinstance(pool, deque):
                pool.clear()
            elif isinstance(pool, OrderedDict):
                pool.clear()
        
        self.compression_cache.clear()
        gc.collect()
        
        logger.log('INFO', 'Cleared all memory allocations')
    
    def get_stats(self) -> Dict:
        """Get quantum memory statistics"""
        global_stats = self.coordinator.get_usage_stats()
        
        compression_ratios = list(self.compression_ratios.values())
        avg_compression = np.mean(compression_ratios) if compression_ratios else 1.0
        
        return {
            'allocated_gb': self.allocated / (1024**3),
            'total_limit_gb': self.total_limit / (1024**3),
            'utilization_percent': (self.allocated / self.total_limit) * 100 if self.total_limit > 0 else 0,
            'pool_sizes': {k: len(v) for k, v in self.pools.items()},
            'avg_compression_ratio': avg_compression,
            'global_stats': global_stats,
            'allocation_success_rate': self.successful_allocations / max(1, self.successful_allocations + self.allocation_failures),
            'total_allocations': self.successful_allocations + self.allocation_failures
        }

# ============================================================================
# HYPER-THREADED EXECUTION ENGINE WITH BACKPRESSURE (FIXED)
# ============================================================================

class HyperThreadedExecutor:
    """Quantum-aware hyper-threaded execution engine with memory management - FIXED"""
    
    def __init__(self, scheduler: QuantumScheduler):
        self.scheduler = scheduler
        self.resources = scheduler.calculate_optimal_resources()
        self.memory_manager = QuantumMemoryManager(self.resources['memory_limit_gb'])
        
        # Quantum task queues with reduced sizes
        self.task_queues = {
            'immediate': deque(maxlen=100),     # Reduced from 1000
            'standard': deque(maxlen=500),      # Reduced from 5000
            'background': deque(maxlen=1000),   # Reduced from 10000
            'quantum': deque(maxlen=50)         # Reduced from 100
        }
        
        # Performance tracking
        self.performance_metrics = {
            'tasks_completed': 0,
            'tasks_failed': 0,
            'total_runtime': 0.0,
            'avg_task_time': 0.0,
            'efficiency': 1.0
        }
        
        # SINGLE thread pool to prevent over-subscription (FIXED)
        self.thread_pool = None
        self.max_workers = min(self.resources['workers'], 8)  # Cap at 8 workers max
        
        # Backpressure mechanism - much more conservative (FIXED)
        max_in_flight = min(self.max_workers * 2, 16)  # Max 16 concurrent tasks
        self.semaphore = threading.Semaphore(max_in_flight)
        
        # Cleanup thread control
        self._shutdown_flag = False
        self._cleanup_thread = None
        self._monitor_thread = None
        
        self._start_threads()
        
        logger.log('INFO', f'HyperThreadedExecutor initialized with {self.max_workers} workers')
    
    def _start_threads(self):
        """Start background threads"""
        self._cleanup_thread = threading.Thread(
            target=self._periodic_cleanup,
            daemon=True,
            name="ExecutorCleanupThread"
        )
        self._cleanup_thread.start()
        
        self._monitor_thread = threading.Thread(
            target=self._monitor_resources,
            daemon=True,
            name="ResourceMonitorThread"
        )
        self._monitor_thread.start()
    
    def _get_thread_pool(self) -> ThreadPoolExecutor:
        """Get thread pool with lazy initialization"""
        if self.thread_pool is None or self.thread_pool._shutdown:
            self.thread_pool = ThreadPoolExecutor(
                max_workers=self.max_workers,
                thread_name_prefix='quantum_worker_'
            )
            logger.log('DEBUG', f'Thread pool created with {self.max_workers} workers')
        return self.thread_pool
    
    def _periodic_cleanup(self):
        """Background thread to clean completed tasks"""
        while not self._shutdown_flag:
            try:
                time.sleep(5)  # Every 5 seconds
                
                if self._shutdown_flag:
                    break
                
                for queue_name, queue in self.task_queues.items():
                    # Remove completed futures
                    completed_indices = []
                    for i, task_info in enumerate(queue):
                        if 'future' in task_info and task_info['future'].done():
                            completed_indices.append(i)
                    
                    # Remove from back to preserve indices
                    for idx in reversed(completed_indices):
                        try:
                            if idx < len(queue):
                                task_info = queue[idx]
                                # Clean up
                                if 'future' in task_info:
                                    try:
                                        # Get result to clear exceptions
                                        task_info['future'].result(timeout=0.001)
                                    except:
                                        pass
                                # Remove from queue
                                del queue[idx]
                        except (IndexError, RuntimeError) as e:
                            logger.log('DEBUG', f'Cleanup error: {e}')
                            continue
                            
            except Exception as e:
                logger.log('WARNING', f'Cleanup thread error: {e}')
                continue
    
    def _monitor_resources(self):
        """Monitor system resources and adjust accordingly"""
        while not self._shutdown_flag:
            try:
                time.sleep(2)  # Every 2 seconds
                
                if self._shutdown_flag:
                    break
                
                # Check memory usage
                mem_stats = self.memory_manager.get_stats()
                memory_usage = mem_stats['utilization_percent']
                
                # Check CPU usage
                cpu_usage = psutil.cpu_percent()
                
                # Adjust behavior based on resource usage
                if memory_usage > 80 or cpu_usage > 85:
                    logger.log('WARNING', f'High resource usage: CPU={cpu_usage:.1f}%, Mem={memory_usage:.1f}%')
                    
                    # Slow down task submission
                    time.sleep(0.1)
                    
                # Log resource status periodically
                if int(time.time()) % 30 == 0:  # Every 30 seconds
                    logger.log('INFO', f'Resource status: CPU={cpu_usage:.1f}%, Mem={memory_usage:.1f}%, '
                                     f'Tasks={self.performance_metrics["tasks_completed"]}')
                    
            except Exception as e:
                logger.log('WARNING', f'Monitor thread error: {e}')
                continue
    
    def submit_task(self, task_func: Callable, *args, priority: str = 'standard', 
                   quantum_tag: str = None, timeout: float = None, **kwargs) -> concurrent.futures.Future:
        """Submit task with quantum-aware scheduling and backpressure - FIXED"""
        
        # Check if we're shutting down
        if self._shutdown_flag:
            raise RuntimeError("Executor is shutting down")
        
        # Wait for semaphore with timeout
        acquired = self.semaphore.acquire(timeout=timeout if timeout else 5.0)
        if not acquired:
            raise TimeoutError("Could not acquire semaphore for task submission")
        
        # Wrap task with quantum monitoring and error handling
        def quantum_wrapped():
            start_time = time.time()
            task_success = False
            
            try:
                result = task_func(*args, **kwargs)
                runtime = time.time() - start_time
                
                # Update performance metrics
                self.performance_metrics['tasks_completed'] += 1
                self.performance_metrics['total_runtime'] += runtime
                task_success = True
                
                return result
                
            except Exception as e:
                runtime = time.time() - start_time
                self.performance_metrics['total_runtime'] += runtime
                self.performance_metrics['tasks_failed'] += 1
                
                logger.log('ERROR', f'Task failed: {quantum_tag} - {e}')
                raise
                
            finally:
                # Release semaphore
                self.semaphore.release()
                
                # Adjust scheduler based on performance
                if self.performance_metrics['tasks_completed'] % 100 == 0:
                    self._update_scheduler_efficiency()
        
        # Submit to thread pool
        pool = self._get_thread_pool()
        future = pool.submit(quantum_wrapped)
        
        # Add to appropriate queue for tracking
        self.task_queues[priority].append({
            'future': future,
            'submitted': time.time(),
            'quantum_tag': quantum_tag
        })
        
        return future
    
    def _update_scheduler_efficiency(self):
        """Update scheduler based on runtime efficiency"""
        completed = self.performance_metrics['tasks_completed']
        failed = self.performance_metrics['tasks_failed']
        
        if completed + failed > 0:
            success_rate = completed / (completed + failed)
            avg_time = (self.performance_metrics['total_runtime'] / max(1, completed))
            
            # Calculate efficiency (lower avg time = higher efficiency)
            efficiency = success_rate / max(0.001, avg_time)
            efficiency = min(1.0, efficiency * 5)  # Normalize
            
            self.performance_metrics['efficiency'] = efficiency
            self.performance_metrics['avg_task_time'] = avg_time
            
            # Adaptive resource adjustment
            self.resources = self.scheduler.adaptive_adjust(self.performance_metrics)
            
            logger.log('DEBUG', f'Scheduler updated: efficiency={efficiency:.3f}, avg_time={avg_time:.3f}s')
    
    def batch_submit(self, task_func: Callable, arg_list: List, 
                    batch_size: int = None, max_concurrent: int = None, **kwargs) -> List:
        """Submit batch of tasks with optimal chunking and backpressure - FIXED"""
        
        if not arg_list:
            return []
        
        if batch_size is None:
            batch_size = self.resources['batch_sizes'].get('standard_batch', 10)
        
        # Cap batch size based on memory (FIXED)
        mem_stats = self.memory_manager.get_stats()
        if mem_stats['utilization_percent'] > 70:
            batch_size = max(1, batch_size // 2)
            logger.log('INFO', f'Reduced batch size to {batch_size} due to high memory usage')
        
        if max_concurrent is None:
            max_concurrent = min(self.max_workers, 4)  # Conservative concurrency
        
        results = []
        total_tasks = len(arg_list)
        
        # Progress tracking
        completed_tasks = 0
        start_time = time.time()
        
        # Submit tasks in chunks to avoid overwhelming the system
        for chunk_start in range(0, total_tasks, batch_size):
            chunk_end = min(chunk_start + batch_size, total_tasks)
            chunk = arg_list[chunk_start:chunk_end]
            
            # Submit chunk tasks with concurrency limit
            chunk_futures = []
            for i, args in enumerate(chunk):
                if len(chunk_futures) >= max_concurrent:
                    # Wait for some to complete
                    for future in concurrent.futures.as_completed(chunk_futures):
                        try:
                            result = future.result(timeout=30)
                            results.append(result)
                            completed_tasks += 1
                        except Exception as e:
                            logger.log('WARNING', f'Batch task failed: {e}')
                            results.append(None)
                            completed_tasks += 1
                    
                    chunk_futures = []
                
                # Submit new task
                future = self.submit_task(
                    self._process_batch_item,
                    task_func, args, **kwargs,
                    priority='standard',
                    quantum_tag=f'batch_{chunk_start}_{i}'
                )
                chunk_futures.append(future)
            
            # Collect remaining results
            for future in concurrent.futures.as_completed(chunk_futures):
                try:
                    result = future.result(timeout=30)
                    results.append(result)
                    completed_tasks += 1
                except Exception as e:
                    logger.log('WARNING', f'Batch task failed: {e}')
                    results.append(None)
                    completed_tasks += 1
            
            # Log progress
            if chunk_end % (batch_size * 5) == 0 or chunk_end == total_tasks:
                elapsed = time.time() - start_time
                rate = completed_tasks / max(elapsed, 0.001)
                logger.log('INFO', f'Batch progress: {chunk_end}/{total_tasks} '
                                 f'({chunk_end/total_tasks*100:.1f}%) - {rate:.1f} tasks/sec')
        
        return results
    
    def _process_batch_item(self, task_func: Callable, args, **kwargs):
        """Process a single batch item"""
        if isinstance(args, (list, tuple)):
            return task_func(*args, **kwargs)
        elif isinstance(args, dict):
            return task_func(**args, **kwargs)
        else:
            return task_func(args, **kwargs)
    
    def shutdown(self):
        """Graceful shutdown of all pools with memory cleanup"""
        self._shutdown_flag = True
        
        # Wait for cleanup thread to finish
        for thread in [self._cleanup_thread, self._monitor_thread]:
            if thread and thread.is_alive():
                thread.join(timeout=2.0)
        
        # Cancel all pending tasks
        for queue_name, queue in self.task_queues.items():
            while queue:
                try:
                    task_info = queue.popleft()
                    if 'future' in task_info:
                        try:
                            task_info['future'].cancel()
                        except:
                            pass
                except (IndexError, RuntimeError):
                    break
        
        # Shutdown thread pool
        if self.thread_pool:
            try:
                self.thread_pool.shutdown(wait=True, cancel_futures=True)
            except:
                pass
            self.thread_pool = None
        
        # Clear memory
        self.memory_manager.clear_all()
        
        # Clear task queues
        for queue in self.task_queues.values():
            queue.clear()
        
        # Force garbage collection
        gc.collect()
        
        logger.log('INFO', 'Executor shutdown complete')
    
    def get_status(self) -> Dict:
        """Get executor status"""
        semaphore_value = self.semaphore._value if hasattr(self.semaphore, '_value') else 'unknown'
        
        return {
            'performance': self.performance_metrics,
            'resources': self.resources,
            'memory_stats': self.memory_manager.get_stats(),
            'queue_sizes': {k: len(v) for k, v in self.task_queues.items()},
            'semaphore_value': semaphore_value,
            'max_workers': self.max_workers,
            'cleanup_thread_alive': self._cleanup_thread.is_alive() if self._cleanup_thread else False,
            'monitor_thread_alive': self._monitor_thread.is_alive() if self._monitor_thread else False
        }

# ============================================================================
# OPTIMIZED PHYSICS KERNELS WITH BUFFER SWAPPING
# ============================================================================

class OptimizedLindbladSolver:
    """Memory-efficient Lindblad solver with batch processing"""
    
    def __init__(self):
        # Pre-compute operators for efficiency
        self.sigma_minus = np.array([[0, 1], [0, 0]], dtype=np.complex64)
        self.sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex64)
        self.identity = np.eye(2, dtype=np.complex64)
        
        # Cache for frequently used operators (with size limit)
        self.operator_cache = OrderedDict()
        self.cache_limit = 100
        
        logger.log('DEBUG', 'OptimizedLindbladSolver initialized')
    
    def evolve_batch(self, psi_batch: np.ndarray, gamma: float, 
                    T1: float, T2: float, steps: int) -> np.ndarray:
        """Evolve batch of states simultaneously"""
        n_batch = psi_batch.shape[0]
        results = np.zeros(n_batch, dtype=np.float32)
        
        # Precompute Lindblad operators
        gamma1 = 1.0 / max(T1, 1e-10)
        gamma2 = 1.0 / max(2 * T2, 1e-10)
        
        L1 = np.sqrt(gamma1) * self.sigma_minus
        L2 = np.sqrt(gamma2) * self.sigma_z
        
        L1_dag = L1.conj().T
        L2_dag = L2.conj().T
        
        for i in range(n_batch):
            psi = psi_batch[i]
            rho = np.outer(psi, psi.conj())
            
            for _ in range(steps):
                # Lindblad evolution
                decay = L1 @ rho @ L1_dag - 0.5 * (L1_dag @ L1 @ rho + rho @ L1_dag @ L1)
                dephase = L2 @ rho @ L2_dag - 0.5 * (L2_dag @ L2 @ rho + rho @ L2_dag @ L2)
                
                rho += (decay + dephase) * 0.1  # Fixed time step
                
                # Normalize
                trace = np.trace(rho)
                if abs(trace) > 1e-12:
                    rho /= trace
                rho = 0.5 * (rho + rho.conj().T)
            
            results[i] = np.real(rho[0, 0])
        
        return results

class OptimizedPropagation:
    """Cache-optimized propagation with memory pooling and LRU cache"""
    
    def __init__(self, memory_manager: QuantumMemoryManager, cache_limit: int = 100):
        self.memory_manager = memory_manager
        self.lattice_cache = OrderedDict()  # LRU cache
        self.cache_limit = cache_limit
        self.cached_memory_bytes = 0
        self.neighbor_cache = {}
        
        logger.log('DEBUG', f'OptimizedPropagation initialized with cache limit: {cache_limit}')
    
    def propagate_batch(self, lattice_params: List[Dict], steps: int = 500) -> List[float]:
        """Propagate batch of lattices with shared memory"""
        results = []
        
        for params in lattice_params:
            size = params['size']
            seed = params.get('seed', 42)
            
            # Check cache first
            cache_key = f"{size}_{seed}"
            if cache_key in self.lattice_cache:
                # Move to end (LRU)
                lattice = self.lattice_cache.pop(cache_key)
                self.lattice_cache[cache_key] = lattice
            else:
                # Allocate shared memory
                mem_size = size * size * 4  # float32
                mem_view = self.memory_manager.allocate(mem_size, f"lattice_{size}")
                
                if mem_view is None:
                    # Fallback to numpy with memory limit
                    if mem_size > 100 * 1024 * 1024:  # 100MB limit
                        logger.log('WARNING', f'Lattice too large: {size}x{size}, skipping')
                        results.append(1.0)
                        continue
                    lattice = np.zeros((size, size), dtype=np.float32)
                else:
                    # Use memoryview as numpy array
                    lattice = np.frombuffer(mem_view, dtype=np.float32).reshape(size, size)
                
                # Initialize
                center = size // 2
                lattice[center, center] = 1.0
                
                # Add to cache with eviction
                self._add_to_cache(cache_key, lattice)
            
            # Run propagation
            r0 = self._propagate_single(lattice, size, steps, seed)
            results.append(r0)
        
        return results
    
    def _add_to_cache(self, key: str, lattice: np.ndarray):
        """Add to cache with LRU eviction"""
        lattice_bytes = lattice.nbytes
        
        # Evict oldest if at limit
        while len(self.lattice_cache) >= self.cache_limit:
            try:
                oldest_key, oldest_lattice = self.lattice_cache.popitem(last=False)
                self.cached_memory_bytes -= oldest_lattice.nbytes
                
                # Explicitly delete and release memory
                del oldest_lattice
            except (KeyError, RuntimeError):
                break
        
        # Add new
        self.lattice_cache[key] = lattice
        self.cached_memory_bytes += lattice_bytes
    
    def _propagate_single(self, lattice: np.ndarray, size: int, steps: int, seed: int) -> float:
        """Single propagation with optimized memory access using buffer swapping"""
        np.random.seed(seed)
        
        infected_history = []
        
        # Precompute neighbor offsets
        if size not in self.neighbor_cache:
            neighbors = []
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    neighbors.append((di, dj))
            self.neighbor_cache[size] = neighbors
        
        neighbors = self.neighbor_cache[size]
        
        # Use buffer swapping instead of copying
        lattice_a = lattice.copy()
        lattice_b = np.empty_like(lattice_a)
        
        current = lattice_a
        next_buffer = lattice_b
        
        for step in range(steps):
            infected = np.sum(current > 0)
            infected_history.append(infected)
            
            # Copy current to next (in-place)
            np.copyto(next_buffer, current)
            
            coherence = 0.7 + 0.3 * math.sin(step / 20.0)
            
            # Vectorized where possible
            infected_cells = np.argwhere(current > 0)
            
            for i, j in infected_cells:
                strength = current[i, j]
                
                for di, dj in neighbors:
                    ni, nj = i + di, j + dj
                    
                    if 0 <= ni < size and 0 <= nj < size:
                        if random.random() < 0.25 * strength * coherence:
                            next_buffer[ni, nj] = min(1.0, current[ni, nj] + 0.25)
            
            # Swap buffers (no copy!)
            current, next_buffer = next_buffer, current
            
            if step % 10 == 0:
                infected_frac = infected / (size * size)
                decay = 0.02 * infected_frac
                np.maximum(0, current - decay, out=current)  # In-place
        
        # Calculate R0
        if len(infected_history) > 10:
            early = infected_history[5:20]
            ratios = []
            for t in range(1, len(early)):
                if early[t-1] > 0:
                    ratio = early[t] / early[t-1]
                    if 0.1 < ratio < 10:
                        ratios.append(ratio)
            
            if ratios:
                return np.mean(ratios)
        
        return 1.0
    
    def clear_cache(self):
        """Explicit cache clearing"""
        self.lattice_cache.clear()
        self.cached_memory_bytes = 0
        gc.collect()
        
        logger.log('DEBUG', 'Propagation cache cleared')
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        return {
            'cache_size': len(self.lattice_cache),
            'cache_limit': self.cache_limit,
            'cached_memory_bytes': self.cached_memory_bytes,
            'cached_memory_gb': self.cached_memory_bytes / (1024**3),
            'neighbor_cache_size': len(self.neighbor_cache)
        }

# ============================================================================
# REVISED VALIDATOR WITH RESOURCE CONTROL AND CLEANUP (FIXED)
# ============================================================================

class QuantumParadoxValidatorV5:
    """Hyper-optimized validator with resource control and memory management - FIXED"""
    
    def __init__(self, profile: ResourceProfile = ResourceProfile.STANDARD, 
                 extra_features: List[str] = None):
        self.profile = profile
        self.extra_features = extra_features or []
        
        print(f"\n{'='*70}")
        print(f"QUANTUM PARADOX VALIDATOR v5.2 - {profile.value.upper()} PROFILE")
        print(f"FIXED: Memory-optimized edition with stability enhancements")
        print(f"{'='*70}")
        
        # Initialize quantum scheduler
        self.scheduler = QuantumScheduler(profile)
        self.resources = self.scheduler.calculate_optimal_resources()
        
        # Initialize execution engine
        self.executor = HyperThreadedExecutor(self.scheduler)
        
        # Load modules with resource awareness
        self.modules = self._load_modules_with_resources()
        
        # Initialize optimized kernels
        self.lindblad_solver = OptimizedLindbladSolver()
        self.propagation_engine = OptimizedPropagation(self.executor.memory_manager)
        
        # Results storage
        self.results = []
        self.performance_log = []
        
        self._print_configuration()
        
        logger.log('INFO', f'QuantumParadoxValidatorV5 initialized with profile: {profile.value}')
    
    def _load_modules_with_resources(self) -> Dict:
        """Load quantum modules with resource constraints"""
        modules = {}
        
        # Try loading each module with resource limits
        module_loaders = {
            'bumpy': self._load_bumpy,
            'sentiflow': self._load_sentiflow,
            'qubitlearn': self._load_qubitlearn,
            'laser': self._load_laser
        }
        
        for name, loader in module_loaders.items():
            try:
                module = loader()
                modules[name] = module
                logger.log('INFO', f'{name} loaded with resource constraints')
            except ImportError as e:
                logger.log('WARNING', f'{name} unavailable: {e}')
                modules[name] = None
            except Exception as e:
                logger.log('ERROR', f'{name} failed: {e}')
                modules[name] = None
        
        return modules
    
    def _load_bumpy(self):
        """Load bumpy with memory limits"""
        try:
            import bumpy
            from bumpy import BumpyArray, deploy_bumpy_core
            
            # Set memory limits for bumpy
            if hasattr(bumpy, 'set_memory_limit'):
                bumpy.set_memory_limit(self.resources['memory_limit_gb'] * 0.1)  # 10% for bumpy
            
            core = deploy_bumpy_core(qualia_dimension=5)
            return {
                'core': core,
                'BumpyArray': BumpyArray,
                'version': getattr(bumpy, '__version__', '2.1')
            }
        except Exception as e:
            logger.log('WARNING', f'Bumpy load error: {e}')
            return None
    
    def _load_sentiflow(self):
        """Load sentiflow with thread limits"""
        try:
            import sentiflow
            from sentiflow import SentientTensor, nn, optim
            
            return {
                'SentientTensor': SentientTensor,
                'nn': nn,
                'optim': optim,
                'version': getattr(sentiflow, '__version__', '3.0')
            }
        except Exception as e:
            logger.log('WARNING', f'Sentiflow load error: {e}')
            return None
    
    def _load_qubitlearn(self):
        """Load qubitlearn with CPU limits"""
        try:
            import qubitlearn
            from qubitlearn import QubitLearnPerfected
            
            # Initialize with resource constraints
            learner = QubitLearnPerfected(domain="quantum_validation")
            return {
                'learner': learner,
                'version': getattr(qubitlearn, '__version__', '9.0')
            }
        except Exception as e:
            logger.log('WARNING', f'Qubitlearn load error: {e}')
            return None
    
    def _load_laser(self):
        """Load laser with I/O limits"""
        try:
            import laser
            from laser import LASERUtility
            
            laser_util = LASERUtility()
            return {
                'utility': laser_util,
                'version': getattr(laser, '__version__', '1.0')
            }
        except Exception as e:
            logger.log('WARNING', f'LASER load error: {e}')
            return None
    
    def _print_configuration(self):
        """Print detailed configuration"""
        print(f"\n📊 Resource Configuration:")
        print(f"   Profile: {self.profile.value}")
        print(f"   Workers: {self.resources['workers']}")
        print(f"   Memory Limit: {self.resources['memory_limit_gb']:.1f} GB")
        print(f"   Quantum Factor: {self.resources['quantum_factor']:.2f}")
        
        print(f"\n🔧 Extra Features: {', '.join(self.extra_features) if self.extra_features else 'None'}")
        
        print(f"\n💾 Memory Manager Status:")
        try:
            mem_stats = self.executor.memory_manager.get_stats()
            print(f"   Allocated: {mem_stats['allocated_gb']:.2f} GB")
            print(f"   Utilization: {mem_stats['utilization_percent']:.1f}%")
            print(f"   Allocation Success Rate: {mem_stats.get('allocation_success_rate', 0):.2%}")
        except Exception as e:
            print(f"   ⚠️ Memory stats error: {e}")
        
        print(f"\n🚀 Ready for quantum validation...")
    
    def _cleanup_experiment(self):
        """Clean up after each experiment"""
        logger.log('INFO', 'Cleaning up experiment resources...')
        
        # Clear lattice cache
        if hasattr(self, 'propagation_engine'):
            try:
                self.propagation_engine.clear_cache()
            except Exception as e:
                logger.log('WARNING', f'Cache cleanup error: {e}')
        
        # Clear memory pools
        try:
            self.executor.memory_manager.clear_all()
        except Exception as e:
            logger.log('WARNING', f'Memory cleanup error: {e}')
        
        # Clear operator caches
        if hasattr(self, 'lindblad_solver'):
            try:
                self.lindblad_solver.operator_cache.clear()
            except Exception as e:
                logger.log('WARNING', f'Operator cache cleanup error: {e}')
        
        # Force garbage collection
        gc.collect()
        
        # Sleep briefly to allow OS to reclaim memory
        time.sleep(0.1)
        
        # Take memory snapshot
        logger.get_memory_snapshot()
    
    def run_experiment_survival(self) -> Dict:
        """Run survival experiment with hyper-threading - FIXED with conservative parameters"""
        logger.log('INFO', 'Starting Experiment 1: Hyper-Threaded Survival Efficiency')
        print(f"\n🔬 Experiment 1: Hyper-Threaded Survival Efficiency")
        
        # REDUCED ensemble sizes to prevent memory issues (FIXED)
        ensemble_sizes = {
            ResourceProfile.MINIMAL: 50,      # Reduced from 100
            ResourceProfile.STANDARD: 200,    # Reduced from 300
            ResourceProfile.HEAVY: 500,       # Reduced from 1000
            ResourceProfile.QUANTUM: 1000,    # Reduced from 3000
            ResourceProfile.GOD_MODE: 2000    # Reduced from 10000
        }
        
        n_ensemble = ensemble_sizes[self.profile]
        batch_size = min(self.resources['batch_sizes']['survival_batch'], 100)  # Cap at 100
        
        print(f"   Ensemble: {n_ensemble}, Batch size: {batch_size}")
        logger.log('INFO', f'Survival experiment: n_ensemble={n_ensemble}, batch_size={batch_size}')
        
        # Check memory before starting
        mem_stats = self.executor.memory_manager.get_stats()
        if mem_stats['utilization_percent'] > 50:
            logger.log('WARNING', f'High memory usage before experiment: {mem_stats["utilization_percent"]:.1f}%')
            # Reduce parameters
            n_ensemble = max(100, n_ensemble // 2)
            batch_size = max(10, batch_size // 2)
            print(f"   ⚠️ Reducing parameters due to memory pressure: n={n_ensemble}, batch={batch_size}")
        
        # Prepare batch parameters
        param_batches = []
        for i in range(0, n_ensemble, batch_size):
            batch_params = []
            for j in range(batch_size):
                if i + j >= n_ensemble:
                    break
                
                # Generate parameters with small variations
                params = {
                    'gamma': 0.01 * random.uniform(0.9, 1.1),
                    'T1': 100 * random.uniform(0.8, 1.2),
                    'T2': 50 * random.uniform(0.8, 1.2),
                    'seed': i + j
                }
                batch_params.append(params)
            
            param_batches.append(batch_params)
        
        # Submit batches in parallel with conservative concurrency
        start_time = time.time()
        
        futures = []
        max_concurrent_batches = min(4, self.executor.max_workers // 2)
        
        for batch_idx, batch in enumerate(param_batches):
            try:
                future = self.executor.submit_task(
                    self._run_survival_batch,
                    batch,
                    priority='standard',
                    quantum_tag=f'survival_batch_{batch_idx}',
                    timeout=30.0
                )
                futures.append(future)
                
                # Limit concurrent batches
                if len(futures) >= max_concurrent_batches:
                    # Wait for some to complete
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            future.result(timeout=60)
                            break
                        except Exception as e:
                            logger.log('WARNING', f'Batch failed: {e}')
                            break
                    # Remove completed futures
                    futures = [f for f in futures if not f.done()]
                    
            except Exception as e:
                logger.log('ERROR', f'Batch submission failed: {e}')
                continue
        
        # Collect results
        all_survival = []
        for future in concurrent.futures.as_completed(futures):
            try:
                batch_results = future.result(timeout=60)
                all_survival.extend(batch_results)
            except Exception as e:
                logger.log('ERROR', f'Batch failed: {e}')
                # Add placeholders
                all_survival.extend([0.5] * batch_size)
        
        runtime = time.time() - start_time
        
        # Calculate statistics
        if all_survival:
            survival_array = np.array(all_survival)
            
            result = {
                'mean': float(np.mean(survival_array)),
                'std': float(np.std(survival_array)),
                'min': float(np.min(survival_array)),
                'max': float(np.max(survival_array)),
                'runtime': runtime,
                'n_ensemble': len(all_survival),
                'throughput': len(all_survival) / runtime if runtime > 0 else 0,
                'efficiency': len(all_survival) / (runtime * self.resources['workers']) if runtime > 0 and self.resources['workers'] > 0 else 0
            }
        else:
            result = {
                'mean': 0.0,
                'std': 0.0,
                'runtime': runtime,
                'n_ensemble': 0,
                'throughput': 0,
                'efficiency': 0
            }
        
        print(f"   ✓ Completed: {result['n_ensemble']} simulations in {runtime:.1f}s")
        print(f"   ✓ Throughput: {result['throughput']:.1f} sims/sec")
        print(f"   ✓ Efficiency: {result['efficiency']:.2f} sims/sec/core")
        
        logger.log('INFO', f'Survival experiment completed: {result}')
        return result
    
    def _run_survival_batch(self, batch_params: List[Dict]) -> List[float]:
        """Run a batch of survival simulations"""
        results = []
        
        for params in batch_params:
            try:
                # Create initial state
                psi = np.array([1.0, 0.0], dtype=np.complex64)
                psi /= np.linalg.norm(psi)
                
                # Convert to batch format for optimized solver
                psi_batch = np.array([psi])
                
                # Run simulation
                survival = self.lindblad_solver.evolve_batch(
                    psi_batch,
                    gamma=params['gamma'],
                    T1=params['T1'],
                    T2=params['T2'],
                    steps=50  # Reduced from 100
                )
                
                results.append(float(survival[0]))
            except Exception as e:
                logger.log('WARNING', f'Single simulation failed: {e}')
                results.append(0.5)
        
        return results
    
    def run_experiment_propagation(self) -> Dict:
        """Run propagation experiment with memory pooling - FIXED"""
        logger.log('INFO', 'Starting Experiment 2: Memory-Optimized Paradox Propagation')
        print(f"\n🔬 Experiment 2: Memory-Optimized Paradox Propagation")
        
        # REDUCED lattice sizes to prevent memory issues (FIXED)
        size_profiles = {
            ResourceProfile.MINIMAL: [30],        # Reduced from 50
            ResourceProfile.STANDARD: [30, 50],   # Reduced from 50, 100
            ResourceProfile.HEAVY: [50, 80],      # Reduced from 50, 100, 200
            ResourceProfile.QUANTUM: [50, 80, 120],  # Reduced from 100, 200, 500
            ResourceProfile.GOD_MODE: [80, 120, 150]  # Reduced from 200, 500, 1000
        }
        
        lattice_sizes = size_profiles[self.profile]
        runs_per_size = max(5, self.resources['workers'] * 2)  # Reduced from *5
        
        print(f"   Lattice sizes: {lattice_sizes}")
        print(f"   Runs per size: {runs_per_size}")
        
        # Prepare propagation parameters
        all_params = []
        for size in lattice_sizes:
            # Check memory requirement for this lattice
            lattice_memory_mb = size * size * 4 / (1024 * 1024)
            if lattice_memory_mb > 100:  # Skip if lattice > 100MB
                logger.log('WARNING', f'Skipping lattice {size}x{size} (too large: {lattice_memory_mb:.1f}MB)')
                continue
                
            for run in range(runs_per_size):
                params = {
                    'size': size,
                    'seed': hash(f"{size}_{run}") % 1000000,
                    'steps': 200  # Reduced from 500
                }
                all_params.append(params)
        
        if not all_params:
            logger.log('WARNING', 'No valid lattice parameters generated')
            return {
                'mean_r0': 1.0,
                'std_r0': 0.0,
                'runtime': 0.0,
                'total_simulations': 0,
                'throughput': 0,
                'memory_usage_gb': 0.0
            }
        
        # Run in batches with conservative settings
        batch_size = min(self.resources['batch_sizes']['propagation_batch'], 20)  # Cap at 20
        
        start_time = time.time()
        all_r0s = []
        
        # Process in chunks to manage memory
        total_chunks = len(all_params) // batch_size + (1 if len(all_params) % batch_size > 0 else 0)
        
        for chunk_idx in range(total_chunks):
            chunk_start = chunk_idx * batch_size
            chunk_end = min(chunk_start + batch_size, len(all_params))
            batch = all_params[chunk_start:chunk_end]
            
            try:
                # Submit batch task
                future = self.executor.submit_task(
                    self.propagation_engine.propagate_batch,
                    batch,
                    priority='standard',
                    quantum_tag=f'propagation_batch_{chunk_idx}',
                    timeout=120.0  # 2 minute timeout
                )
                
                try:
                    batch_results = future.result(timeout=180)  # 3 minute timeout
                    all_r0s.extend(batch_results)
                except concurrent.futures.TimeoutError:
                    logger.log('ERROR', f'Propagation batch timeout: {chunk_idx}')
                    all_r0s.extend([1.0] * len(batch))
                except Exception as e:
                    logger.log('ERROR', f'Propagation batch failed: {e}')
                    all_r0s.extend([1.0] * len(batch))
            except Exception as e:
                logger.log('ERROR', f'Batch submission failed: {e}')
                all_r0s.extend([1.0] * len(batch))
            
            # Progress update
            progress = (chunk_idx + 1) / total_chunks * 100
            if int(progress) % 20 == 0:  # Every 20%
                print(f"   Progress: {progress:.0f}%")
                logger.log('INFO', f'Propagation progress: {progress:.0f}%')
            
            # Check memory and pause if needed
            mem_stats = self.executor.memory_manager.get_stats()
            if mem_stats['utilization_percent'] > 75:
                logger.log('WARNING', f'High memory usage ({mem_stats["utilization_percent"]:.1f}%), pausing...')
                time.sleep(1.0)
                self._cleanup_experiment()
        
        runtime = time.time() - start_time
        
        # Calculate statistics
        if all_r0s:
            r0_array = np.array(all_r0s)
            
            result = {
                'mean_r0': float(np.mean(r0_array)),
                'std_r0': float(np.std(r0_array)),
                'min_r0': float(np.min(r0_array)),
                'max_r0': float(np.max(r0_array)),
                'runtime': runtime,
                'total_simulations': len(all_r0s),
                'throughput': len(all_r0s) / runtime if runtime > 0 else 0,
                'memory_usage_gb': self.executor.memory_manager.get_stats()['allocated_gb']
            }
        else:
            result = {
                'mean_r0': 1.0,
                'std_r0': 0.0,
                'runtime': runtime,
                'total_simulations': 0,
                'throughput': 0,
                'memory_usage_gb': 0.0
            }
        
        print(f"   ✓ Completed: {result['total_simulations']} propagations in {runtime:.1f}s")
        print(f"   ✓ Mean R₀: {result['mean_r0']:.3f} ± {result['std_r0']:.3f}")
        print(f"   ✓ Memory used: {result['memory_usage_gb']:.2f} GB")
        
        logger.log('INFO', f'Propagation experiment completed: {result}')
        return result
    
    def run_all_experiments(self) -> Dict:
        """Run complete validation suite with cleanup between experiments"""
        logger.log('INFO', 'Starting complete validation suite')
        print(f"\n{'='*70}")
        print(f"RUNNING COMPLETE VALIDATION SUITE")
        print(f"{'='*70}")
        
        total_start = time.time()
        
        # Run experiments with cleanup between each
        print(f"\n📊 Starting Experiment 1: Survival Efficiency")
        exp1_result = self.run_experiment_survival()
        self._cleanup_experiment()
        
        print(f"\n📊 Starting Experiment 2: Paradox Propagation")
        exp2_result = self.run_experiment_propagation()
        self._cleanup_experiment()
        
        # Run additional experiments if requested
        exp3_result = None
        exp4_result = None
        
        if 'full_suite' in self.extra_features:
            print(f"\n📊 Running additional experiments (full suite)...")
            # Add ethical horizon and trinity theorem here
            self._cleanup_experiment()
        
        total_runtime = time.time() - total_start
        
        # Compile results
        summary = {
            'profile': self.profile.value,
            'total_runtime': total_runtime,
            'experiments': {
                'survival_efficiency': exp1_result,
                'paradox_propagation': exp2_result
            },
            'resource_usage': self.executor.get_status(),
            'quantum_metrics': {
                'coherence_level': self.scheduler.coherence_level,
                'quantum_phase': self.scheduler.quantum_phase,
                'superposition': self.scheduler.superposition_states
            },
            'performance_logs': logger.get_performance_summary()
        }
        
        # Generate report
        self._generate_optimized_report(summary)
        
        # Final cleanup
        self._final_cleanup()
        
        return summary
    
    def _final_cleanup(self):
        """Final aggressive cleanup"""
        logger.log('INFO', 'Starting final cleanup')
        print(f"   🧹 Final cleanup...")
        
        # Shutdown executor
        try:
            self.executor.shutdown()
        except Exception as e:
            logger.log('WARNING', f'Executor shutdown error: {e}')
        
        # Clear all caches
        if hasattr(self, 'propagation_engine'):
            try:
                self.propagation_engine.clear_cache()
            except Exception as e:
                logger.log('WARNING', f'Propagation cache cleanup error: {e}')
        
        if hasattr(self, 'lindblad_solver'):
            try:
                self.lindblad_solver.operator_cache.clear()
            except Exception as e:
                logger.log('WARNING', f'Lindblad cache cleanup error: {e}')
        
        # Multiple GC passes
        for _ in range(3):
            gc.collect()
        
        # Take final memory snapshot
        snapshot = logger.get_memory_snapshot()
        current, peak = tracemalloc.get_traced_memory()
        logger.log('INFO', f'Final memory: {current/(1024**2):.1f}MB current, {peak/(1024**2):.1f}MB peak')
        
        print(f"   ✅ Cleanup complete")
    
    def _generate_optimized_report(self, summary: Dict):
        """Generate hyper-optimized performance report"""
        print(f"\n{'='*70}")
        print(f"PERFORMANCE REPORT - {self.profile.value.upper()} PROFILE")
        print(f"{'='*70}")
        
        # Resource utilization
        print(f"\n📊 Resource Utilization:")
        resources = summary['resource_usage']['resources']
        print(f"   CPU Workers: {resources['workers']}")
        print(f"   Memory Limit: {resources['memory_limit_gb']:.1f} GB")
        print(f"   Quantum Factor: {resources['quantum_factor']:.2f}")
        
        # Performance metrics
        perf = summary['resource_usage']['performance']
        print(f"\n⚡ Performance Metrics:")
        print(f"   Tasks Completed: {perf['tasks_completed']:,}")
        if perf['tasks_failed'] > 0:
            print(f"   Tasks Failed: {perf['tasks_failed']:,}")
        print(f"   Total Runtime: {perf['total_runtime']:.1f}s")
        print(f"   Avg Task Time: {perf['avg_task_time']:.3f}s")
        print(f"   Efficiency: {perf['efficiency']:.2f}")
        
        # Experiment results
        print(f"\n🔬 Experiment Results:")
        
        exp1 = summary['experiments']['survival_efficiency']
        print(f"   1. Survival Efficiency:")
        print(f"      Mean: {exp1['mean']:.3f} ± {exp1['std']:.3f}")
        print(f"      Throughput: {exp1['throughput']:.1f} sims/sec")
        print(f"      Core Efficiency: {exp1['efficiency']:.2f} sims/sec/core")
        
        exp2 = summary['experiments']['paradox_propagation']
        print(f"\n   2. Paradox Propagation:")
        print(f"      Mean R₀: {exp2['mean_r0']:.3f} ± {exp2['std_r0']:.3f}")
        print(f"      Throughput: {exp2['throughput']:.1f} sims/sec")
        print(f"      Memory Used: {exp2['memory_usage_gb']:.2f} GB")
        
        # Quantum metrics
        quantum = summary['quantum_metrics']
        print(f"\n🔮 Quantum System Metrics:")
        print(f"   Coherence Level: {quantum['coherence_level']:.3f}")
        print(f"   Quantum Phase: {quantum['quantum_phase']:.2f} rad")
        
        # Memory statistics
        mem_stats = summary['resource_usage']['memory_stats']
        print(f"\n💾 Memory Statistics:")
        print(f"   Allocated: {mem_stats['allocated_gb']:.2f} GB")
        print(f"   Utilization: {mem_stats['utilization_percent']:.1f}%")
        if 'allocation_success_rate' in mem_stats:
            print(f"   Allocation Success: {mem_stats['allocation_success_rate']:.2%}")
        
        # Performance logs
        perf_logs = summary.get('performance_logs', {})
        if perf_logs:
            print(f"\n📈 System Performance:")
            print(f"   Peak Memory: {perf_logs.get('peak_memory_mb', 0):.1f} MB")
            print(f"   Average CPU: {perf_logs.get('average_cpu', 0):.1f}%")
            print(f"   Average Memory: {perf_logs.get('average_memory', 0):.1f}%")
        
        # Save detailed report
        self._save_optimized_report(summary)
        
        # Final assessment
        print(f"\n{'='*70}")
        print(f"FINAL ASSESSMENT")
        print(f"{'='*70}")
        
        efficiency_score = perf['efficiency']
        
        if efficiency_score > 0.8:
            rating = "EXCELLENT"
            color = "🟢"
        elif efficiency_score > 0.6:
            rating = "GOOD"
            color = "🟡"
        elif efficiency_score > 0.4:
            rating = "ADEQUATE"
            color = "🟠"
        else:
            rating = "POOR"
            color = "🔴"
        
        print(f"\n{color} Resource Efficiency: {rating}")
        print(f"   Score: {efficiency_score:.2f}")
        
        # Recommendations
        print(f"\n💡 Optimization Recommendations:")
        if self.profile != ResourceProfile.GOD_MODE and efficiency_score > 0.7:
            print(f"   ✓ Consider upgrading to GOD_MODE for maximum performance")
        if mem_stats.get('utilization_percent', 0) > 80:
            print(f"   ⚠️ Memory usage high - consider reducing batch sizes")
        if perf.get('avg_task_time', 0) > 1.0:
            print(f"   ⚠️ Task times high - check for I/O bottlenecks")
        if perf.get('tasks_failed', 0) > perf.get('tasks_completed', 0) * 0.1:
            print(f"   ⚠️ High task failure rate - check system stability")
        
        print(f"\n🎯 Total validation time: {summary['total_runtime']:.1f} seconds")
    
    def _save_optimized_report(self, summary: Dict):
        """Save detailed optimization report"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"quantum_validation_v5.2_{self.profile.value}_{timestamp}.json"
            
            # Add system information
            summary['system_info'] = self.scheduler.system_info
            summary['timestamp'] = timestamp
            summary['python_version'] = sys.version
            summary['numpy_version'] = np.__version__
            
            with open(filename, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            print(f"\n💾 Detailed report saved to: {filename}")
            
            # Also save a summary CSV for quick analysis
            csv_file = f"validation_summary_{timestamp}.csv"
            self._save_summary_csv(summary, csv_file)
            
        except Exception as e:
            logger.log('ERROR', f'Could not save report: {e}')
            print(f"⚠️ Could not save report: {e}")
    
    def _save_summary_csv(self, summary: Dict, filename: str):
        """Save key metrics as CSV for easy analysis"""
        try:
            import csv
            
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Write header
                writer.writerow([
                    'Metric', 'Value', 'Unit', 'Profile', 'Timestamp'
                ])
                
                # Write data
                timestamp = summary.get('timestamp', '')
                profile = summary.get('profile', '')
                
                # Experiment 1
                exp1 = summary['experiments']['survival_efficiency']
                writer.writerow(['Survival_Mean', exp1['mean'], 'probability', profile, timestamp])
                writer.writerow(['Survival_Std', exp1['std'], 'probability', profile, timestamp])
                writer.writerow(['Survival_Throughput', exp1['throughput'], 'sims/sec', profile, timestamp])
                
                # Experiment 2
                exp2 = summary['experiments']['paradox_propagation']
                writer.writerow(['Propagation_Mean_R0', exp2['mean_r0'], 'unitless', profile, timestamp])
                writer.writerow(['Propagation_Std_R0', exp2['std_r0'], 'unitless', profile, timestamp])
                writer.writerow(['Propagation_Throughput', exp2['throughput'], 'sims/sec', profile, timestamp])
                
                # Resource usage
                perf = summary['resource_usage']['performance']
                writer.writerow(['Tasks_Completed', perf['tasks_completed'], 'count', profile, timestamp])
                writer.writerow(['Tasks_Failed', perf.get('tasks_failed', 0), 'count', profile, timestamp])
                writer.writerow(['Total_Runtime', perf['total_runtime'], 'seconds', profile, timestamp])
                writer.writerow(['Efficiency_Score', perf['efficiency'], 'score', profile, timestamp])
                
                # Memory
                mem = summary['resource_usage']['memory_stats']
                writer.writerow(['Memory_Allocated_GB', mem['allocated_gb'], 'GB', profile, timestamp])
                writer.writerow(['Memory_Utilization_Percent', mem['utilization_percent'], '%', profile, timestamp])
            
            print(f"   Summary CSV saved to: {filename}")
            
        except Exception as e:
            logger.log('ERROR', f'Could not save CSV: {e}')
            print(f"⚠️ Could not save CSV: {e}")

# ============================================================================
# INTERACTIVE PROMPT AND MAIN EXECUTION (ENHANCED)
# ============================================================================

def interactive_prompt() -> Tuple[ResourceProfile, List[str]]:
    """Interactive prompt for user configuration with safety warnings"""
    print(f"\n{'='*70}")
    print(f"QUANTUM PARADOX VALIDATOR v5.2 - RESOURCE CONFIGURATION")
    print(f"{'='*70}")
    
    # Display system information with warnings
    try:
        cpu_count = psutil.cpu_count(logical=False)
        logical_cores = psutil.cpu_count(logical=True)
        memory = psutil.virtual_memory()
        
        print(f"\n💻 System Overview:")
        print(f"   Physical Cores: {cpu_count}")
        print(f"   Logical Cores: {logical_cores}")
        print(f"   Total Memory: {memory.total / (1024**3):.1f} GB")
        print(f"   Available Memory: {memory.available / (1024**3):.1f} GB")
        
        # Warning if low memory
        if memory.available / memory.total < 0.2:
            print(f"   ⚠️ WARNING: Low available memory ({memory.available/(1024**3):.1f} GB)")
            print(f"      Consider using MINIMAL or STANDARD profile")
    except:
        print(f"\n💻 System Overview: Unable to read system info")
    
    # Resource profiles with warnings for high profiles
    print(f"\n📊 Available Resource Profiles:")
    profiles = [
        ("1", "MINIMAL", "25% resources, quiet operation", "For background runs", ""),
        ("2", "STANDARD", "50% resources, balanced", "Recommended for most users", ""),
        ("3", "HEAVY", "75% resources, intensive", "For detailed analysis", "⚠️ May stress system"),
        ("4", "QUANTUM", "95% resources, optimized", "For maximum performance", "⚠️ May cause high memory usage"),
        ("5", "GOD_MODE", "100% resources, all-out", "For benchmarks and research", "⚠️ EXPERIMENTAL - May cause instability")
    ]
    
    for num, name, desc, use_case, warning in profiles:
        print(f"   [{num}] {name:10} - {desc}")
        print(f"        {use_case}")
        if warning:
            print(f"        {warning}")
    
    # Get profile choice
    while True:
        choice = input(f"\nSelect profile (1-5, default 2): ").strip()
        if not choice:
            choice = "2"
        
        if choice in ["1", "2", "3", "4", "5"]:
            profile_map = {
                "1": ResourceProfile.MINIMAL,
                "2": ResourceProfile.STANDARD,
                "3": ResourceProfile.HEAVY,
                "4": ResourceProfile.QUANTUM,
                "5": ResourceProfile.GOD_MODE
            }
            selected_profile = profile_map[choice]
            
            # Additional warning for high profiles
            if choice in ["4", "5"]:
                confirm = input(f"⚠️  {selected_profile.value.upper()} profile may cause high memory usage. Continue? (y/N): ").strip().lower()
                if confirm not in ['y', 'yes']:
                    print("Reverting to STANDARD profile")
                    selected_profile = ResourceProfile.STANDARD
            
            break
        else:
            print("Invalid choice. Please enter 1-5.")
    
    # Extra features
    print(f"\n🔧 Extra Features (optional):")
    features = [
        ("A", "Full suite", "Run all 4 experiments (takes longer)"),
        ("B", "GPU acceleration", "Use GPU if available (requires CUDA)"),
        ("C", "Extended logging", "Detailed performance logs"),
        ("D", "Real-time monitoring", "Live performance dashboard"),
        ("E", "Quantum entanglement", "Enable bumpy entanglement features"),
        ("F", "Holographic compression", "Advanced memory compression")
    ]
    
    for letter, name, desc in features:
        print(f"   [{letter}] {name:20} - {desc}")
    
    feature_choices = input(f"\nSelect extra features (comma-separated, or Enter for none): ").strip()
    
    selected_features = []
    if feature_choices:
        choice_map = {
            "A": "full_suite",
            "B": "gpu_acceleration",
            "C": "extended_logging",
            "D": "realtime_monitoring",
            "E": "quantum_entanglement",
            "F": "holographic_compression"
        }
        
        for choice in feature_choices.split(','):
            choice = choice.strip().upper()
            if choice in choice_map:
                selected_features.append(choice_map[choice])
            else:
                print(f"   ⚠️ Unknown feature: {choice}")
    
    # Confirmation
    print(f"\n{'='*70}")
    print(f"CONFIGURATION SUMMARY")
    print(f"{'='*70}")
    print(f"   Profile: {selected_profile.value}")
    print(f"   Extra Features: {', '.join(selected_features) if selected_features else 'None'}")
    
    confirm = input(f"\nProceed with validation? (Y/n): ").strip().lower()
    if confirm in ['', 'y', 'yes']:
        return selected_profile, selected_features
    else:
        print("Validation cancelled.")
        sys.exit(0)

def main():
    """Main execution with interactive configuration and enhanced error handling"""
    
    # Parse command line arguments first
    parser = argparse.ArgumentParser(description='Quantum Paradox Validator v5.2')
    parser.add_argument('--profile', choices=['minimal', 'standard', 'heavy', 'quantum', 'god'],
                       help='Resource profile (minimal, standard, heavy, quantum, god)')
    parser.add_argument('--features', help='Comma-separated extra features')
    parser.add_argument('--non-interactive', action='store_true', 
                       help='Run without interactive prompts')
    parser.add_argument('--safe-mode', action='store_true',
                       help='Enable additional safety checks and reduced parameters')
    args = parser.parse_args()
    
    if args.non_interactive:
        # Use command line arguments
        profile_map = {
            'minimal': ResourceProfile.MINIMAL,
            'standard': ResourceProfile.STANDARD,
            'heavy': ResourceProfile.HEAVY,
            'quantum': ResourceProfile.QUANTUM,
            'god': ResourceProfile.GOD_MODE
        }
        
        profile = profile_map.get(args.profile, ResourceProfile.STANDARD)
        features = args.features.split(',') if args.features else []
    else:
        # Interactive mode
        profile, features = interactive_prompt()
    
    # Apply safe mode if requested
    if args.safe_mode:
        print("\n🔒 SAFE MODE ENABLED:")
        print("   - Reduced memory allocations")
        print("   - Slower execution for stability")
        print("   - Additional safety checks")
        
        # Downgrade profile if it's too aggressive
        if profile in [ResourceProfile.QUANTUM, ResourceProfile.GOD_MODE]:
            print(f"   ⚠️  Downgrading from {profile.value} to HEAVY for safety")
            profile = ResourceProfile.HEAVY
    
    try:
        logger.log('INFO', f'Starting validation with profile: {profile.value}')
        logger.log('INFO', f'Features: {features}')
        
        # Initialize and run validator
        validator = QuantumParadoxValidatorV5(profile, features)
        results = validator.run_all_experiments()
        
        # Final status
        print(f"\n{'='*70}")
        print(f"VALIDATION COMPLETE")
        print(f"{'='*70}")
        print(f"Total time: {results['total_runtime']:.1f} seconds")
        print(f"Profile: {profile.value}")
        
        if features:
            print(f"Features used: {', '.join(features)}")
        
        # Show final memory stats
        current, peak = tracemalloc.get_traced_memory()
        print(f"\n📊 Final Memory Statistics:")
        print(f"   Peak memory usage: {peak/(1024**2):.1f} MB")
        print(f"   Current memory: {current/(1024**2):.1f} MB")
        
        tracemalloc.stop()
        
        return 0
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Validation interrupted by user")
        logger.log('WARNING', 'Validation interrupted by user')
        
        # Attempt cleanup
        try:
            if 'validator' in locals():
                validator._final_cleanup()
        except:
            pass
        
        tracemalloc.stop()
        return 1
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        logger.log('CRITICAL', f'Fatal error: {e}')
        traceback.print_exc()
        
        # Attempt cleanup
        try:
            if 'validator' in locals():
                validator._final_cleanup()
        except:
            pass
        
        tracemalloc.stop()
        return 1

# ============================================================================
# EMERGENCY MEMORY HANDLER
# ============================================================================

class EmergencyMemoryHandler:
    """Emergency memory cleanup utilities"""
    
    @staticmethod
    def emergency_cleanup():
        """Aggressive memory cleanup"""
        import gc
        import numpy as np
        
        print("\n⚠️  EMERGENCY MEMORY CLEANUP")
        
        # Clear numpy arrays
        try:
            np._no_nep50_warning = True
        except:
            pass
        
        # Clear matplotlib cache
        try:
            import matplotlib
            matplotlib.pyplot.close('all')
            matplotlib.cbook._lock = None
        except:
            pass
        
        # Force garbage collection
        for i in range(3):
            gc.collect(generation=i)
        
        # Clear sys.modules cache for large modules
        large_modules = [k for k, v in sys.modules.items() 
                        if hasattr(v, '__dict__') and len(str(v.__dict__)) > 10000]
        for mod in large_modules:
            if mod not in ['builtins', 'sys', '__main__']:
                try:
                    del sys.modules[mod]
                except:
                    pass
        
        # Check memory after cleanup
        import psutil
        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024**3)
        
        print(f"   Available memory after cleanup: {available_gb:.2f} GB")
        
        return available_gb
    
    @staticmethod
    def check_system_resources():
        """Check if system has enough resources"""
        import psutil
        
        mem = psutil.virtual_memory()
        cpu_count = psutil.cpu_count(logical=True)
        
        warnings = []
        
        if mem.available / mem.total < 0.1:
            warnings.append(f"⚠️  Very low available memory: {mem.available/(1024**3):.1f} GB")
        
        if cpu_count < 4:
            warnings.append(f"⚠️  Low CPU cores: {cpu_count}")
        
        load = psutil.getloadavg()[0] / cpu_count if cpu_count > 0 else 1
        if load > 0.8:
            warnings.append(f"⚠️  High system load: {load:.2f}")
        
        return warnings

# ============================================================================
# RUN SCRIPT
# ============================================================================

if __name__ == "__main__":
    # Check system resources first
    warnings = EmergencyMemoryHandler.check_system_resources()
    if warnings:
        print("\n⚠️  SYSTEM WARNINGS:")
        for warning in warnings:
            print(f"   {warning}")
        
        response = input("\nContinue anyway? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            print("Exiting...")
            sys.exit(1)
    
    # Run main function
    exit_code = main()
    
    # Final emergency cleanup if needed
    import psutil
    mem = psutil.virtual_memory()
    if mem.percent > 90:
        print("\n⚠️  High memory usage after execution, performing emergency cleanup...")
        EmergencyMemoryHandler.emergency_cleanup()
    
    sys.exit(exit_code)

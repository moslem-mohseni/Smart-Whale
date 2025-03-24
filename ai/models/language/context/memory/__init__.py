from .l1_cache import L1Cache
from .l2_cache import L2Cache
from .l3_cache import L3Cache
from .cache_synchronizer import CacheSynchronizer
from .quantum_compressor import QuantumCompressor
from .quantum_memory import QuantumMemory
from .retrieval_optimizer import RetrievalOptimizer


l1_cache = L1Cache()
l2_cache = L2Cache()
l3_cache = L3Cache()
cache_synchronizer = CacheSynchronizer()
quantum_memory = QuantumMemory()
quantum_compressor = QuantumCompressor()
retrieval_optimizer = RetrievalOptimizer()


__all__ = [
    "L1Cache",
    "L2Cache",
    "L3Cache",
    "CacheSynchronizer",
    "QuantumCompressor",
    "QuantumMemory",
    "RetrievalOptimizer"
]

"""Domain-aware KV-cache compression library.

Exported modules:
- section_parser: parse document sections
- allocator: compute per-section budgets
- compressor: compression/eviction/quantization routines
- kvpress_adapter: adapter hooks for connecting to KVPress/vLLM
"""

from .section_parser import extract_sections
from .allocator import allocate_budgets
from .compressor import compress_kv_cache
from .kvpress_adapter import KVPressAdapter

__all__ = [
    "extract_sections",
    "allocate_budgets",
    "compress_kv_cache",
    "KVPressAdapter",
]

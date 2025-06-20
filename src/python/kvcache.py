from typing import List
import numpy as np
from pyo3 import wrap_pyfunction, create_exception
# KVCacheException is a custom exception for KV cache operations.
create_exception = create_exception if 'create_exception' in globals() else None

wrap_pyfunction = wrap_pyfunction if 'wrap_pyfunction' in globals() else None

# KVCacheException is a custom exception for KV cache operations.
class KVCacheException(Exception):
    """Custom exception for KV cache operations."""
    if create_exception:
        KVCacheException = create_exception("KVCacheException", Exception)

        if wrap_pyfunction:
            @wrap_pyfunction
            def kv_cache_exception(message: str):
                raise KVCacheException(message)

        else:
            def __init__(self, message: str):
                super().__init__(message)
            def __str__(self):
                return f"KVCacheException: {self.args[0]}"
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message
    def __str__(self):
        return f"KVCacheException: {self.message}"

@wrap_pyfunction
def kv_cache_exception(message: str):
    raise KVCacheException(message)

class Profiler:
    def __init__(self):
        self.profile_data = {}

    def profile(self, layer_name: str, func):
        import time
        start_time = time.time()
        result = func()
        elapsed_time = time.time() - start_time
        self.profile_data[layer_name] = elapsed_time
        return result

    def get_profile(self):
        return self.profile_data

## KVCacheManager is a Python class that manages the KV cache for a neural network model.
class KVCacheManager:
    def __init__(self):
        self.current_precision = "int8"

    def update_precision(self, attention_data: List[float], system_load: float):
        # Call Rust via pyo3 or similar FFI
        pass

    def deduplicate_and_quantize(self, attention_data: List[float]):
        pass

    def _quantize_fp8(self, data: List[float]) -> List[float]:
        # Approximate FP8 quantization
        return [np.clip(x * 255.0, -128.0, 127.0) / 255.0 for x in data]

    def _quantize_int4(self, data: List[float]) -> List[float]:
        # Approximate INT4 quantization
        return [np.clip(x * 15.0, -8.0, 7.0) / 15.0 for x in data]

    def _quantize_int8(self, data: List[float]) -> List[float]:
        # Approximate INT8 quantization
        return [np.clip(x * 127.0, -128.0, 127.0) / 127.0 for x in data]

    def _quantize_bf16(self, data: List[float]) -> List[float]:
        # Approximate BF16 quantization
        return [np.float16(x) for x in data]

    def _quantize_float16(self, data: List[float]) -> List[float]:
        # Approximate FP16 quantization
        return [np.float16(x) for x in data]

    def _quantize_float32(self, data: List[float]) -> List[float]:
        # No quantization for FP32
        return data

    def _quantize(self, data: List[float], precision: str) -> List[float]:
        if precision == "fp8":
            return self._quantize_fp8(data)
        elif precision == "int4":
            return self._quantize_int4(data)
        elif precision == "int8":
            return self._quantize_int8(data)
        elif precision == "bf16":
            return self._quantize_bf16(data)
        elif precision == "float16":
            return self._quantize_float16(data)
        elif precision == "float32":
            return self._quantize_float32(data)
        else:
            raise KVCacheException(f"Unsupported precision: {precision}")

    def process_attention_data(self, attention_data: List[float], precision: str) -> List[float]:
        if precision not in ["fp8", "int4", "int8", "bf16", "float16", "float32"]:
            raise KVCacheException(f"Unsupported precision: {precision}")

        # Deduplicate and quantize the attention data
        deduplicated_data = list(set(attention_data))
        quantized_data = self._quantize(deduplicated_data, precision)
        
        return quantized_data

    def generate_ir(self, layer_data: List[dict]):
        ir_data = []
        for layer in layer_data:
            if 'data' not in layer or 'precision' not in layer:
                raise KVCacheException("Layer data must contain 'data' and 'precision' keys")
            
            quantized_data = self._quantize(layer['data'], layer['precision'])
            ir_data.append({
                'name': layer.get('name', 'unknown'),
                'data': quantized_data,
                'precision': layer['precision']
            })
        
        return ir_data

        



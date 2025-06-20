import numpy as np
from typing import List
import time
from pyo3 import wrap_pyfunction, create_exception

class MermaidFlowException(Exception):
    pass

@wrap_pyfunction
def mermaid_flow_exception(message: str):
    raise MermaidFlowException(message)

class MermaidFlow:
    def __init__(self):
        self.precision_plan = PrecisionPlan()

    def process_flow(self, input_tokens: List[float]) -> List[float]:
        profiler = Profiler()
        output = profiler.profile("embedding", lambda: self.embedding_layer(input_tokens))
        output = profiler.profile("attention_qkv", lambda: self.attention_qkv(output, "int4"))
        output = profiler.profile("sparse_attention", lambda: self.sparse_attention(output))
        output = profiler.profile("feed_forward", lambda: self.feed_forward(output))
        output = profiler.profile("output_fp8", lambda: self.output_layer_fp8(output))
        print("Profiling Results:", profiler.get_profile())
        return output

    def embedding_layer(self, input_tokens: List[float]) -> List[float]:
        return [self._quantize_fp8(x * 0.1) for x in input_tokens]  # FP8 embedding

    def attention_qkv(self, input_data: List[float], precision: str) -> List[float]:
        return input_data  # Placeholder for INT4

    def sparse_attention(self, input_data: List[float]) -> List[float]:
        return input_data  # Placeholder for INT4/INT8

    def feed_forward(self, input_data: List[float]) -> List[float]:
        return input_data  # Placeholder for INT8/BF16

    def output_layer_fp8(self, input_data: List[float]) -> List[float]:
        return [self._quantize_fp8(x) for x in input_data]  # FP8 output

    def _quantize_fp8(self, value: float) -> float:
        # Approximate FP8 quantization
        scaled = np.clip(value * 255.0, -128.0, 127.0)
        return scaled / 255.0

class PrecisionPlan:
    def __init__(self):
        self.profiler = Profiler()

    def compute_plan(self, layer_data: List[float]) -> dict:
        entropy = self._calculate_entropy(layer_data)
        sensitivity = self._estimate_sensitivity(layer_data)
        profile = self.profiler.profile("profile_layer", lambda: self._profile_layer(layer_data))

        precision = "float16" if entropy > 0.8 or sensitivity > 0.5 else \
                    "int4" if profile["load"] < 0.3 else "fp8"

        return {"precision": precision, "sparsity": 0.9 if entropy < 0.2 else 0.3}

    def _calculate_entropy(self, data: List[float]) -> float:
        return sum(x * x for x in data) if data else 0.0

    def _estimate_sensitivity(self, data: List[float]) -> float:
        return 0.4  # Placeholder

    def _profile_layer(self, data: List[float]) -> dict:
        return {"load": 0.5}  # Placeholder

class Profiler:
    def __init__(self):
        self.times = {}

    def profile(self, name: str, func):
        start = time.time()
        result = func()
        self.times[name] = time.time() - start
        return result

    def get_profile(self):
        return self.times

        
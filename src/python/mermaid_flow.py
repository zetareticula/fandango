import numpy as np
from typing import List

class MermaidFlow:
    def __init__(self):
        self.precision_plan = PrecisionPlan()

    def process_flow(self, input_tokens: List[float]) -> List[float]:
        output = self.embedding_layer(input_tokens)
        output = self.attention_qkv(output, "int4")
        output = self.sparse_attention(output)
        output = self.feed_forward(output)
        return self.output_layer_fp8(output)

    def embedding_layer(self, input_tokens: List[float]) -> List[float]:
        return [x * 0.1 for x in input_tokens]  # FP8 embedding

    def attention_qkv(self, input_data: List[float], precision: str) -> List[float]:
        return input_data  # Placeholder for INT4

    def sparse_attention(self, input_data: List[float]) -> List[float]:
        return input_data  # Placeholder for INT4/INT8

    def feed_forward(self, input_data: List[float]) -> List[float]:
        return input_data  # Placeholder for INT8/BF16

    def output_layer_fp8(self, input_data: List[float]) -> List[float]:
        # Simple FP8 quantization
        return [(round(x * 255.0) / 255.0) for x in input_data]

class PrecisionPlan:
    def compute_plan(self, layer_data: List[float]) -> dict:
        entropy = self._calculate_entropy(layer_data)
        sensitivity = self._estimate_sensitivity(layer_data)
        profile = self._profile_layer(layer_data)

        precision = "float16" if entropy > 0.8 or sensitivity > 0.5 else \
                    "int4" if profile["load"] < 0.3 else "fp8" if layer_data.is_empty() else "int8"

        return {"precision": precision, "sparsity": 0.9 if entropy < 0.2 else 0.3}

    def _calculate_entropy(self, data: List[float]) -> float:
        return sum(x * x for x in data) if data else 0.0

    def _estimate_sensitivity(self, data: List[float]) -> float:
        return 0.4  # Placeholder

    def _profile_layer(self, data: List[float]) -> dict:
        return {"load": 0.5}  # Placeholder
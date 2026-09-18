from sglang.multi_model.uma.adapters.dense_decoder import DenseDecoderAdapter


class Qwen3Adapter(DenseDecoderAdapter):
    architectures = frozenset({"Qwen3ForCausalLM"})

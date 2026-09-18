from sglang.multi_model.uma.adapters.dense_decoder import DenseDecoderAdapter


class LlamaFamilyAdapter(DenseDecoderAdapter):
    architectures = frozenset(
        {
            "LlamaForCausalLM",
            "MistralForCausalLM",
        }
    )

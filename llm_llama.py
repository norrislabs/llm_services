from llm_base import BaseLanguageModel

from langchain_community.llms import LlamaCpp


class LlamaModel(BaseLanguageModel):
    def __init__(self, model, template, verbose, **kwargs):
        super().__init__(model, template, verbose, **kwargs)

    def create_llm(self, model, verbose, kwargs):
        model_info = {'model': model.split('/').pop(),
                      'model_type': 'llama',
                      'temperature': kwargs.pop('temperature', 0.0),
                      'max_tokens': kwargs.pop('max_tokens', 1024),
                      'n_ctx': kwargs.pop('n_ctx', 2048),
                      'gpu_layers': kwargs.pop('gpu', 0)
                      }

        # Fire up the Llama 2 based LLM
        # LangChain
        llm = LlamaCpp(
            model_path=model,
            temperature=model_info['temperature'],
            max_tokens=model_info['max_tokens'],
            n_ctx=model_info['n_ctx'],
            n_gpu_layers=model_info['gpu_layers'],
            top_p=1,
            n_batch=512,
            verbose=verbose,
            repeat_penalty=1.2
        )

        return llm, model_info

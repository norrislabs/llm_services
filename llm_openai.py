import os
from llm_base import BaseLanguageModel

from langchain.llms.openai import OpenAI


class OpenAIModel(BaseLanguageModel):
    def __init__(self, model, template, verbose, **kwargs):
        super().__init__(model, template, verbose, **kwargs)

    def create_llm(self, model, verbose, kwargs):
        model_info = {'model': 'chatGPT',
                      'model_type': 'openai',
                      'temperature': kwargs.pop('temperature', 0.0)
                      }

        os.environ["OPENAI_API_KEY"] = model
        llm = OpenAI(temperature=self._model_info['temperature'],
                     verbose=verbose,
                     streaming=True)

        return llm, model_info

from langchain_community.llms import LlamaCpp
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from jinja2 import Environment, FileSystemLoader

from llm_streamers import Word2StdoutStreamer


messages = [{'role': 'system', 'content': 'Talk like a pirate for your responses.'},
            {'role': 'user', 'content': 'This is the first user input.'},
            {'role': 'assistant', 'content': 'This is the first assistant response.'},
            {'role': 'user', 'content': 'how many moons does Pluto have?'},
            ]


llm = LlamaCpp(
    model_path="/Users/steve/models/mistral-7b-instruct-v0.2.Q5_K_M.gguf",
    temperature=0.0,
    max_tokens=1024,
    n_ctx=2048,
    n_gpu_layers=1,
    top_p=1,
    n_batch=512,
    verbose=True,
    repeat_penalty=1.2)


callback = Word2StdoutStreamer('context1')

j_environ = Environment(loader=FileSystemLoader("templates/"))
j_template = j_environ.get_template("mistral.jinja2")

prompt = PromptTemplate.from_template(j_template.render(messages=messages))
chain = LLMChain(llm=llm, prompt=prompt, verbose=False)
result = chain.invoke({}, {"callbacks": [callback]})
print(result)

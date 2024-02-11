import pyfiglet
import argparse
import logging
import asyncio
from pydantic import BaseModel, typing, Field

from fastapi.responses import StreamingResponse
from fastapi import FastAPI, HTTPException, status
import uvicorn

from llm_streamers import Word2QueueStreamer
from llm_llama import LlamaModel
from llm_openai import OpenAIModel

llm_types = {'llama': LlamaModel,
             'openai': OpenAIModel
             }


class Predict(BaseModel):
    msg: str


class ContextSpec(BaseModel):
    template: str = Field(default='')
    history: int = Field(default=0)
    system_prompt: str = Field(default='')
    summerizer_type: str = Field(default='')


class ReturnData(BaseModel):
    name: str
    detail: typing.Any


# creating a fast application
app = FastAPI()


@app.post("/llm/restart")
async def restart_llm() -> ReturnData:
    app.extra['llm'].restart()
    return ReturnData(name="llm", detail='LLM restarted')


@app.post("/llm/shutdown")
async def shutdown_llm() -> ReturnData:
    app.extra['llm'].shutdown()
    return ReturnData(name="llm", detail='LLM shutdown')


@app.get("/llm/list")
async def list_contexts() -> ReturnData:
    names = app.extra['llm'].get_context_names()
    return ReturnData(name="llm", detail=names)


@app.get("/llm/info")
async def model_info() -> ReturnData:
    info = app.extra['llm'].model_info
    return ReturnData(name="llm", detail=info)


@app.post("/context/{name}")
def create_context(name: str, cspec: ContextSpec) -> ReturnData:
    if app.extra['llm'].create_context(name,
                                       template_file=cspec.template,
                                       history_count=cspec.history,
                                       system_prompt=cspec.system_prompt,
                                       summerizer_type=cspec.summerizer_type,
                                       streamer_type='queue'):
        return ReturnData(name=name, detail="Context '{}' created with history of {}".format(name,
                                                                                             cspec.history))
    else:
        return ReturnData(name=name, detail="Reusing context '{}'".format(name))


@app.delete("/context/{name}")
def delete_context(name: str) -> ReturnData:
    if app.extra['llm'].delete_context(name):
        return ReturnData(name=name, detail="Context '{}' deleted".format(name))
    else:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail="Context '{}' does not exist".format(name))


@app.put("/context/{name}")
def submit_directive(name: str, predict: Predict) -> ReturnData:
    resp_id = app.extra['llm'].submit_directive(name, predict.msg)
    if resp_id:
        return ReturnData(name=name, detail=resp_id)
    else:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail="Context '{}' does not exist".format(name))


@app.get('/context/{name}')
async def stream_response(name: str):
    # We use Streaming Response class of Fast API to stream response
    if app.extra['llm'].get_context(name):
        return StreamingResponse(serve_response(app.extra['llm'].get_context(name).streamer),
                                 media_type='text/event-stream')
    else:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail="Context '{}' does not exist".format(name))


@app.get('/context/info/{name}')
async def get_context_info(name: str):
    info = app.extra['llm'].get_context_info(name)
    if info is None:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail="Context '{}' info does not exist".format(name))
    else:
        return ReturnData(name=name, detail=info)


@app.put("/context/template/{name}")
def load_template(name: str, cspec: ContextSpec) -> ReturnData:
    if app.extra['llm'].load_template(name, cspec.template):
        return ReturnData(name=name, detail="Context '{}' template loaded".format(name))
    else:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail="Context '{}' and/or template {} do not exist".format(name, cspec.template))


@app.get("/context/template/{name}")
def get_template(name: str) -> ReturnData:
    templ = app.extra['llm'].get_template(name)
    if templ:
        return ReturnData(name=name, detail=templ)
    else:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail="Template for context '{}' has not been loaded or rendered.".format(name))


@app.put("/context/prompt/{name}")
def set_system_prompt(name: str, cspec: ContextSpec) -> ReturnData:
    if app.extra['llm'].set_system_prompt(name, cspec.system_prompt):
        return ReturnData(name=name, detail="System prompt in context '{}' set.".format(name))
    else:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail="Context '{}' does not exist".format(name))


@app.patch("/context/history/{name}")
def clear_context(name: str) -> ReturnData:
    if app.extra['llm'].clear_context(name):
        return ReturnData(name=name, detail="Context '{}' history cleared".format(name))
    else:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail="Context '{}' does not exist".format(name))


@app.get("/context/history/{name}")
def get_context_history(name: str) -> ReturnData:
    hist = app.extra['llm'].get_history(name)
    if hist:
        return ReturnData(name=name, detail=hist)
    else:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail="Context '{}' does not exist".format(name))


async def serve_response(streamer: Word2QueueStreamer):
    while True:
        # Retreiving the word from the queue
        word = streamer.get_word()

        # yields the value
        yield word

        # provides a task_done signal once value yielded
        streamer.processed_word()

        # Breaks if an end marker is encountered
        if word.startswith('|END'):
            break

        # guard to make sure we are not extracting anything from
        # empty queue
        await asyncio.sleep(0.01)


if __name__ == "__main__":
    print(pyfiglet.figlet_format("LLM Server"))

    logging.basicConfig(level=logging.INFO)

    ap = argparse.ArgumentParser()
    ap.add_argument("llm_type", default="llama", help="LLM type (llama, openai)")
    ap.add_argument("template", default="", help="template file")
    ap.add_argument('model', default="", help="model file or API key")
    ap.add_argument("-p", "--port", type=int, default=8080, help="server port")
    ap.add_argument("-g", "--gpu", type=int, default=0, help="number of gpus")
    ap.add_argument("-t", "--temperature", type=float, default=0.0, help="model temperature (0-1.0)")
    ap.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    ap.add_argument("-c", "--n_ctx", type=int, default=2048, help="size of context")
    ap.add_argument("-m", "--tokens", type=int, default=1024, help="max tokens")
    args = vars(ap.parse_args())

    # Build the model and pass it into the web server
    app.extra['llm'] = llm_types[args["llm_type"]](args["model"],
                                                   args['template'],
                                                   verbose=args["verbose"],
                                                   gpu=args['gpu'],
                                                   temperature=args['temperature'],
                                                   n_ctx=args['n_ctx'],
                                                   max_tokens=args['tokens'])

    # Start the web server
    uvicorn.run(app, host='0.0.0.0', port=args['port'], log_level='info')

    app.extra['llm'].shutdown()

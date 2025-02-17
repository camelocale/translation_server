"""
NOTE: This API server is used only for demonstrating usage of AsyncEngine
and simple performance benchmarks. It is not intended for production use.
For production use, we recommend using our OpenAI compatible server.
We are also not going to accept PRs modifying this file, please
change `vllm/entrypoints/openai/api_server.py` instead.
"""
import pandas as pd
import argparse
import json
import ssl
from typing import AsyncGenerator
from zh_sentence.tokenizer import tokenize

import logging

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.usage.usage_lib import UsageContext
from vllm.utils import random_uuid
from vllm import LLM

from transformers import pipeline

TIMEOUT_KEEP_ALIVE = 5  # seconds.
app = FastAPI()
engine = None

logger = logging.getLogger('uvicorn.error')

lang = "Chinese"

def get_prompt(sents, src_lang):
    sent_list = []
    for i in range(len(sents)):
        prompt = f"""<|im_start|>system
        You are a veteran translator who translates Chinese into Korean. Translate naturally only using Korean.<|im_end|>
        <|im_start|>user
        Translate the following text from Chinese to Korean
        {src_lang}: {sents[i]}
        Korean:<|im_end|>
        <|im_start|>assistant"""

        sent_list.append(str(prompt))
    return sent_list

@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@app.post("/generate")
async def generate(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    request_dict = await request.json()
    prompt = request_dict.pop("prompt")

    stream = request_dict.pop("stream", False)
    sampling_params = SamplingParams(**request_dict)
    request_id = random_uuid()

    sents = tokenize(prompt)
    p = tuple(get_prompt(sents, lang))
    logger.debug(p)

    # stream = True

    assert engine is not None
    results_generator = engine.generate(p, sampling_params, request_id)

    # Streaming case
    async def stream_results() -> AsyncGenerator[bytes, None]:
        async for request_output in results_generator:
            text_outputs = [
                output.text for output in request_output.outputs  #prompt + output.text for output in request_output.outputs
            ]
            ret = {"text": text_outputs}
            yield (json.dumps(ret) + "\0").encode("utf-8")


    if stream:
        return StreamingResponse(stream_results())

    # Non-streaming case
    final_output = None
    async for request_output in results_generator:
        if await request.is_disconnected():
            # Abort the request if the client disconnects.
            await engine.abort(request_id)
            return Response(status_code=499)
        final_output = request_output
        
    assert final_output is not None
    text_outputs = [output.text for output in final_output.outputs] #prompt + output.text for output in final_output.outputs]

    ###
    outputs = " ".join(text_outputs)
    ret = {"text": outputs}
    
    return JSONResponse(ret)



# @app.post("/translate")
# async def translate(request: Request) -> Response:
#     """Translate for the request.

#     The request should be a JSON object with the following fields:
#     - prompt: the prompt to use for the generation.
#     - stream: whether to stream the results or not.
#     - other fields: the sampling parameters (See `SamplingParams` for details).
#     """
#     request_dict = await request.json()
#     prompt = request_dict.pop("prompt")

#     stream = request_dict.pop("stream", False)
#     sampling_params = SamplingParams(**request_dict)
#     # request_id = random_uuid()

#     sents = tokenize(prompt)
#     p = get_prompt(sents, lang)
    
#     text_outputs = llm.generate(p, sampling_params)
#     logger.debug(text_outputs)
#     outputs = " ".join(text_outputs)

#     ret = {"text": outputs}
#     return JSONResponse(ret)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--ssl-keyfile", type=str, default=None)
    parser.add_argument("--ssl-certfile", type=str, default=None)
    parser.add_argument("--ssl-ca-certs",
                        type=str,
                        default=None,
                        help="The CA certificates file")
    parser.add_argument(
        "--ssl-cert-reqs",
        type=int,
        default=int(ssl.CERT_NONE),
        help="Whether client certificate is required (see stdlib ssl module's)"
    )
    parser.add_argument(
        "--root-path",
        type=str,
        default=None,
        help="FastAPI root_path when app is behind a path based routing proxy")
    parser.add_argument("--log-level", type=str, default="debug")
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()
    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(
        engine_args, usage_context=UsageContext.API_SERVER)

    app.root_path = args.root_path
    app
    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level=args.log_level,
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
                ssl_keyfile=args.ssl_keyfile,
                ssl_certfile=args.ssl_certfile,
                ssl_ca_certs=args.ssl_ca_certs,
                ssl_cert_reqs=args.ssl_cert_reqs)

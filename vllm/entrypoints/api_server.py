"""
NOTE: This API server is used only for demonstrating usage of AsyncEngine
and simple performance benchmarks. It is not intended for production use.
For production use, we recommend using our OpenAI compatible server.
We are also not going to accept PRs modifying this file, please
change `vllm/entrypoints/openai/api_server.py` instead.
"""

import argparse
import json
import ssl
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.usage.usage_lib import UsageContext
from vllm.utils import random_uuid

from nltk import sent_tokenize
import fasttext
import re
import uuid
import logging

model = fasttext.load_model('/home/user/Desktop/server/vllm/entrypoints/openai/lid.176.ftz')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TIMEOUT_KEEP_ALIVE = 5  # seconds.
app = FastAPI()
engine = None


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
    src_lang = request_dict.pop("src_lang")
    if src_lang == "Auto":
        src_lang = False
    tgt_lang = request_dict.pop("tgt_lang")

    sampling_params = SamplingParams(
        max_tokens=2048,
        temperature= 0.8,
        stop= '<|im_end|>'
        )
    request_id = random_uuid()

    ## 구현 ##
    def lang_tag(lang):
        if lang == "__label__zh":
            return "Chinese"
        elif lang == "__label__ru":
            return "Russian"
        elif lang == "__label__ko":
            return "Korean"
        elif lang == "__label__es":
            return "Spanish"
        elif lang == "__label__fr":
            return "French"
        elif lang == "__label__de":
            return "German"
        elif lang == "__label__it":
            return "Italian"
        elif lang == "__label__nl":
            return "Dutch"
        elif lang == "__label__pt":
            return "Portuguese"
        elif lang == "__label__ja":
            return "Japanese"
        else:
            return "English"

    def detect_lang(text):
        detected_lang = model.predict(text.replace("\n", ""))[0][0]
        language = lang_tag(detected_lang)
        return language

    def preprocess_text(prompt):
        if detect_lang(prompt) == "Chinese" or "Japanese":
            # prompt = re.sub("[\s]", "", prompt)
            prompt = prompt.replace("，", ", ")
            prompt = prompt.replace("、", ", ")
            prompt = prompt.replace("？", "?")
            prompt = prompt.replace("！", "!")
            prompt = prompt.replace("：", ":")
            prompt = prompt.replace("“", ' "')
            prompt = prompt.replace("”", '" ')
            prompt = prompt.replace("‘", "'")
            prompt = prompt.replace("’", "'")
            prompt = prompt.replace('。"', '。" ')
            prompt = prompt.replace("。'", "。' ")
            prompt = prompt.replace("。", ". ")
            prompt = prompt.replace('. "', '."')
            prompt = prompt.replace(". '", ".'")
            return prompt
        else: 
            return prompt

    def para_tokenize(prompt): 
        tokenized_para = []
        split= prompt.split("\n")
        # print(split)
        buf = ""
        first = True
        for i in range(len(split)):
            split_text = split[i].strip()
            if first and split_text=="" and tokenized_para!=0:
                if len(buf) < 48:
                    if buf != "":
                        if buf[-1] in [".", ",", "!", "?"]:
                            pass
                        else:
                            buf += "."
                    else: pass
                else:
                    tokenized_para.append(buf.strip())
                    buf = ""
                first = False
            elif split_text!="":
                if buf != "":
                    if buf[-1] in [".", ",", "!", "?"]:
                        buf += split_text
                    else:
                        buf += "." + split_text
                else:
                    buf = split_text
                first = True
        if buf != "":
            tokenized_para.append(buf)

        return tokenized_para


    ## 문단 나누는 알고리즘 고도화하기
    def sentence_split(prompt):
        split_list = []
        buf = ""
        len_buf = 0
        preprocess_prompt = preprocess_text(prompt)
        tokenized_para = para_tokenize(preprocess_prompt)
        # print(tokenized_para)

        for j in range(len(tokenized_para)):
            tokenized_sent = sent_tokenize(tokenized_para[j])
            for i, e in enumerate(tokenized_sent):
                if len(e) >= 128:
                    if buf != "" and len(split_list) != 0:
                        prev_text = split_list.pop(-1)
                        if len(prev_text) < len(e):
                            prev_added = prev_text + buf
                            split_list.append(prev_added)
                            buf = ""
                            len_buf = 0
                        else:
                            present_added = buf + e
                            split_list.append(present_added)
                            buf = ""
                            len_buf = 0
                    else: 
                        if buf != "":
                            split_list.append(buf + e)
                            buf = ""
                            len_buf = 0
                        else:
                            split_list.append(e)
                            buf = ""
                            len_buf = 0
                else:
                    buf += e
                    len_buf += len(e)

                    if len_buf < 128:
                        if i == len(tokenized_sent)-1:
                            split_list.append(buf)
                            buf = ""
                            len_buf = 0
                        else:
                            pass
                    else:
                        split_list.append(buf)
                        buf = ""
                        len_buf = 0
        return split_list

    assert engine is not None

    def make_generator(prompt): 
        unique_request_id = str(uuid.uuid4()) 
        results_generator = engine.generate(prompt, sampling_params, unique_request_id)
        return results_generator

    def create_prompt(og_prompt, src_lang, tgt_lang):
        if src_lang:
            system_message = f"You are a professional translator from {src_lang} to {tgt_lang}."
            user_message = f"Translate following text from {src_lang} to {tgt_lang} without omitting or changing the original meaning."
            prompt = f"<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n"
            return prompt
        else:
            src_lang = detect_lang(og_prompt)
            system_message = f"You are a professional translator from {src_lang} to {tgt_lang}."
            user_message = f"Translate following text from {src_lang} to {tgt_lang} without omitting or changing the original meaning."
            prompt = f"<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n"
            return prompt

    def update_user_prompt(prompt, split_text_list, i, src_lang, tgt_lang):
        src_text = split_text_list[i]
        if src_lang:
            if i == 0:
                prompt += f"<|im_start|>user\n{src_text}<|im_end|>\n<|im_start|>assistant\n"
                return prompt
            else:
                user_context_message = f"Considering the context of translations made so far, Translate following text from {src_lang} to {tgt_lang} without omitting or changing the original meaning."
                prompt += f"<|im_start|>user\n{user_context_message}<|im_end|>\n<|im_start|>user\n{src_text}<|im_end|>\n<|im_start|>assistant\n"
                return prompt

        else:
            src_lang = detect_lang(split_text_list[i])
            if i == 0:
                prompt += f"<|im_start|>user\n{src_text}<|im_end|>\n<|im_start|>assistant\n"
                return prompt
            else:
                user_context_message = f"Considering the context of translations made so far, Translate following text from {src_lang} to {tgt_lang} without omitting or changing the original meaning."
                prompt += f"<|im_start|>user\n{user_context_message}<|im_end|>\n<|im_start|>user\n{src_text}<|im_end|>\n<|im_start|>assistant\n"
                return prompt

    def update_assistant_prompt(prompt, result):
        prompt += result + "<|im_end|>\n"
        return prompt

    async def stream_results(og_prompt, src_lang, tgt_lang) -> AsyncGenerator[bytes, None]:
        try:
            split_text_list = sentence_split(og_prompt)
            print(split_text_list)
            prompt = create_prompt(split_text_list[0], src_lang, tgt_lang)
            for i in range(len(split_text_list)):
                len_text = 0
                prompt = update_user_prompt(prompt, split_text_list, i, src_lang, tgt_lang)
                results_generator = make_generator(prompt)
                len_text = 0
                async for request_output in results_generator:
                    text_outputs = [
                        output.text.strip() for output in request_output.outputs
                    ]
                    print(text_outputs[0][len_text:], end="", flush=True)
                    len_text = len(text_outputs[0])
                    yield text_outputs[0]
                prompt = update_assistant_prompt(prompt, text_outputs[0])
        except Exception as e:
            print(f"Error result: {e}")
            yield json.dumps({"text":[]})

    ########

    if stream:
        return StreamingResponse(stream_results(prompt, src_lang, tgt_lang))

    # Non-streaming case
    else:
        split_text_list = sentence_split(prompt)
        prompt = create_prompt(split_text_list[0], src_lang, tgt_lang)
        ret_result = ""
        for i in range(len(split_text_list)):
            prompt = update_user_prompt(prompt, split_text_list, i, src_lang, tgt_lang)
            results_generator = make_generator(prompt)
            final_output = None
            async for request_output in results_generator:
                final_output = request_output
            text_outputs = "".join([output.text.strip() for output in final_output.outputs])
            prompt = update_assistant_prompt(prompt, text_outputs)
            ret_result += text_outputs
        ret = ret_result
        return JSONResponse(ret)


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
    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level=args.log_level,
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
                ssl_keyfile=args.ssl_keyfile,
                ssl_certfile=args.ssl_certfile,
                ssl_ca_certs=args.ssl_ca_certs,
                ssl_cert_reqs=args.ssl_cert_reqs)

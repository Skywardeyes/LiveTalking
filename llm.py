import time
import os
from basereal import BaseReal
from logger import logger

def llm_response(message, nerfreal: BaseReal):
    start = time.perf_counter()
    
    # === 连接本地 Ollama 的 OpenAI 兼容接口 ===
    from openai import OpenAI
    client = OpenAI(
        base_url="http://localhost:11434/v1",  # Ollama 的 OpenAI API 地址
        api_key="ollama"  # Ollama 不需要真实 key，但 SDK 要求填写，任意字符串即可
    )
    end = time.perf_counter()
    logger.info(f"llm Time init: {end - start}s")

    completion = client.chat.completions.create(
        model="qwen3:latest",  # 必须与你本地 pull 的模型名一致
        messages=[
            {'role': 'system', 'content': '你是智能助手，请根据用户的问题给出回答，回答不要超过一百字，不要带任何标点符号，不要使用任何表情符号，不要使用中文以外的任何语言'},
            {'role': 'user', 'content': message}
        ],
        stream=True,
        # 注意：Ollama 不支持 stream_options.include_usage，必须移除
    )

    result = ""
    first = True
    for chunk in completion:
        if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content is not None:
            msg = chunk.choices[0].delta.content
            if first:
                end = time.perf_counter()
                logger.info(f"llm Time to first chunk: {end - start}s")
                first = False

            lastpos = 0
            for i, char in enumerate(msg):
                if char in ",.!;:，。！？：；":
                    segment = result + msg[lastpos:i+1]
                    lastpos = i + 1
                    if len(segment) > 10:
                        logger.info(segment)
                        nerfreal.put_msg_txt(segment)
                        result = ""
                    else:
                        result = segment
            result += msg[lastpos:]

    if result:
        end = time.perf_counter()
        logger.info(f"llm Time to last chunk: {end - start}s")
        nerfreal.put_msg_txt(result)
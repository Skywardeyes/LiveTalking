import os
import time
from urllib.parse import urljoin

from basereal import BaseReal
from logger import logger

try:
    from openai import OpenAI  # type: ignore
except ImportError:  # pragma: no cover
    OpenAI = None


def _build_anythingllm_client():
    """
    通过 OpenAI 兼容接口连接 AnythingLLM。
    需要配置:
        ANYTHINGLLM_BASE_URL -> 例如 http://localhost:3001/api
        ANYTHINGLLM_API_KEY  -> 控制台生成的 key
        ANYTHINGLLM_MODEL    -> workspace slug
    """
    if OpenAI is None:
        raise RuntimeError("openai package not installed")

    base_url = os.getenv("ANYTHINGLLM_BASE_URL", "http://localhost:3001/api")
    api_key = os.getenv("ANYTHINGLLM_API_KEY")
    model = os.getenv("ANYTHINGLLM_MODEL", "3c22d8a4-f57b-41c1-985f-fe13860e3a51")

    if not api_key:
        raise RuntimeError("ANYTHINGLLM_API_KEY not set")

    normalized_base = base_url.rstrip("/") + "/"
    openai_base = urljoin(normalized_base, "v1/openai")

    client = OpenAI(api_key=api_key, base_url=openai_base)
    return client, model


def llm_response(message, nerfreal: BaseReal):
    start = time.perf_counter()

    try:
        client, model = _build_anythingllm_client()
    except Exception:
        logger.exception("init anythingllm client failed")
        nerfreal.put_msg_txt("抱歉，大模型服务暂时不可用。")
        return

    end = time.perf_counter()
    logger.info("llm Time init: %ss", end - start)

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": message},
            ],
            stream=True,
            stream_options={"include_usage": True},
        )
    except Exception:
        logger.exception("anythingllm chat request failed")
        nerfreal.put_msg_txt("抱歉，大模型服务暂时不可用。")
        return

    result = ""
    first = True
    for chunk in completion:
        if len(chunk.choices) == 0:
            continue

        if first:
            end = time.perf_counter()
            logger.info("llm Time to first chunk: %ss", end - start)
            first = False

        msg = chunk.choices[0].delta.content or ""
        lastpos = 0
        for i, char in enumerate(msg):
            if char in ",.!;:，。！？：；":
                result += msg[lastpos : i + 1]
                lastpos = i + 1
                if len(result) > 10:
                    nerfreal.put_msg_txt(result)
                    result = ""
        result += msg[lastpos:]

    end = time.perf_counter()
    logger.info("llm Time to last chunk: %ss", end - start)
    if result:
        nerfreal.put_msg_txt(result)
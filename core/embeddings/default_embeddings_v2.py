import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Any, List, Tuple

import tiktoken
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import run_in_executor
from langchain_openai import OpenAIEmbeddings
from langchain_openai.embeddings.base import _process_batched_chunked_embeddings

from api_routers.context import get_user_id
from configs.model_config import EMBEDDING_MODELS, DEFAULT_EMBEDDING_MODEL, logger
from core.callbacks.llm_callbacks import EmbeddingsTokenUsageCallbackFunction
from core.common.common_constant import PLACEHOLDER_STRING


class OpenAICompatibleEmbeddings(OpenAIEmbeddings):
    embedding_model_id: Optional[str] = None
    batch_size: int = 3  # 批量执行协程任务的数量
    sub_inputs_batch_size: int = 10  # 每个inputs子批次的大小，通易千问限制为10
    user_id: str = None
    place_holder_embedding: Optional[List[float]] = None

    def _record_token_usage(self, total_tokens: int) -> None:
        logger.debug(f"[OpenAICompatibleEmbeddings]-_record_token_usage- {self.user_id} total tokens used: {total_tokens}")
        token_usage_callback_function = EmbeddingsTokenUsageCallbackFunction(
            user_id=self.user_id,
            product_id=self.embedding_model_id,
            product_type=DEFAULT_EMBEDDING_MODEL['product_type'],
        )
        token_usage_callback_function.request_record_token_usage(
            prompt_tokens=total_tokens,
            completion_tokens=0,
            tokens=total_tokens,
        )

    def _tiktoken_tokenize_with_decode(
            self,
            texts: List[str],
            chunk_size: Optional[int] = None,
    ) -> Tuple[List[List[int]], List[str], List[int]]:
        model_name = self.tiktoken_model_name or self.model
        try:
            encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")

        encoder_kwargs: dict[str, Any] = {
            k: v
            for k, v in {
                "allowed_special": self.allowed_special,
                "disallowed_special": self.disallowed_special,
            }.items()
            if v is not None
        }

        token_chunks: List[List[int]] = []
        decoded_chunks: List[str] = []
        indices: List[int] = []

        for i, text in enumerate(texts):
            if self.model.endswith("001"):
                text = text.replace("\n", " ")
            if encoder_kwargs:
                full_tokens = encoding.encode(text, **encoder_kwargs)
            else:
                full_tokens = encoding.encode_ordinary(text)

            for j in range(0, len(full_tokens), self.embedding_ctx_length):
                chunk_tokens = full_tokens[j: j + self.embedding_ctx_length]
                token_chunks.append(chunk_tokens)
                decode_str = encoding.decode(chunk_tokens)
                if chunk_size is not None and len(decode_str) > chunk_size:
                    decode_str = decode_str[:chunk_size]
                    logger.info(
                        f"[OpenAICompatibleEmbeddings]-_tiktoken_tokenize_with_decode-Text chunk at index {i} truncated to {chunk_size} characters."
                        f"The text raw size is {len(text)} and token size is {len(full_tokens)}")
                decoded_chunks.append(decode_str)
                indices.append(i)

        return token_chunks, decoded_chunks, indices

    def __placeholder_embedding_init(self, client_kwargs: Optional[dict]) -> list[float]:
        if self.place_holder_embedding is None:
            try:
                average_embedded = self.client.create(input=PLACEHOLDER_STRING,
                                                      extra_headers=self.headers,
                                                      **client_kwargs)
                if not isinstance(average_embedded, dict):
                    average_embedded = average_embedded.model_dump()
                self.place_holder_embedding = average_embedded["data"][0]["embedding"]

                if (total_tokens := average_embedded.get("usage", {}).get("total_tokens", 0)) > 0:
                    self._record_token_usage(total_tokens)
            except Exception as e:
                logger.error(f"[_placeholder_embedding_init]-Error initializing placeholder embedding: {e}")
        return self.place_holder_embedding

    async def _process_embedding_sub_batch_api_call(self, sub_batch: list[str],
                                              client_kwargs: Optional[dict]) -> list[list[float]]:
        batch_embeddings = []
        try:
            # 使用异步客户端
            response = self.client.create(
                input=sub_batch,
                extra_headers=self.headers,
                **client_kwargs,
            )
            if not isinstance(response, dict):
                response = response.model_dump()
        except Exception as e:
            logger.error(f"[_process_embedding_sub_batch_api_call_async]-Error processing sub-batch {sub_batch}: {e}.\n"
                         f"And will return placeholder embedding or empty list.", exc_info=True)
            # 出现异常时，返回占位符嵌入或空列表
            if self.place_holder_embedding is not None:
                batch_embeddings = [self.place_holder_embedding for _ in sub_batch]
            else:
                placeholder_embedding = self.__placeholder_embedding_init(client_kwargs)
                if placeholder_embedding is not None:
                    batch_embeddings = [placeholder_embedding for _ in sub_batch]
                else:
                    batch_embeddings = [[] for _ in sub_batch]
            return batch_embeddings  # dim: len(sub_batch) * embedding_size


        if (total_tokens := response.get("usage", {}).get("total_tokens", 0)) > 0:
            self._record_token_usage(total_tokens)

        batch_embeddings.extend(r["embedding"] for r in response["data"])
        return batch_embeddings  # dim: len(sub_batch) * embedding_size

    async def _process_all_batches_embedding_async(self, decoded_chunks: list[str],
                                                   client_kwargs: Optional[dict]) -> list[list[float]]:
        tasks = []
        for sub_batch_index in range(0, len(decoded_chunks), self.sub_inputs_batch_size):
            sub_batch = decoded_chunks[sub_batch_index: sub_batch_index + self.sub_inputs_batch_size]
            logger.debug(
                f"[_process_all_batches_embedding_async]-Processing batch from {sub_batch_index} to {sub_batch_index + len(sub_batch)}, "
                f"batch size: {len(sub_batch)}, batch inputs:{sub_batch}")
            task = self._process_embedding_sub_batch_api_call(sub_batch, client_kwargs)
            tasks.append(task)

        # 分批次执行协程任务，每批最多batch_size个
        batch_results = []
        for i in range(0, len(tasks), self.batch_size):
            batch_tasks = tasks[i:i + self.batch_size]
            # 执行当前批次的任务
            current_batch_results = await asyncio.gather(
                *(asyncio.wait_for(task, timeout=30) for task in batch_tasks)
            )
            logger.debug(
                f"[_process_all_batches_embedding_async]-Completed batch {i // self.batch_size + 1} / {(len(tasks) + self.batch_size - 1) // self.batch_size}."
                f"The tasks size:{len(tasks)} and result size: {len(current_batch_results)}")
            batch_results.extend(current_batch_results)

        # 合并所有批次的结果
        batched_embeddings = []
        for batch_result in batch_results:
            batched_embeddings.extend(batch_result)
        return batched_embeddings

    def _get_len_safe_embeddings(
            self,
            texts: list[str],
            *,
            engine: str,
            chunk_size: Optional[int] = None,
            **kwargs: Any,
    ) -> list[list[float]]:
        _chunk_size = chunk_size or self.chunk_size
        client_kwargs = {**self._invocation_params, **kwargs}

        # 显式：使用 tiktoken -> tokens(list[list[int]]) + decoded(list[str])，而不是返回list[int]类型的tokens，主要考虑兼容性问题
        token_chunks, decoded_chunks, indices = self._tiktoken_tokenize_with_decode(texts, chunk_size)

        def run_async_in_thread():
            return asyncio.run(self._process_all_batches_embedding_async(decoded_chunks, client_kwargs))

        with ThreadPoolExecutor() as executor:
            future = executor.submit(run_async_in_thread)
            batched_embeddings = future.result()

        embeddings = _process_batched_chunked_embeddings(
            len(texts),
            token_chunks,  # 使用原始 token 序列以确保加权按 token 数
            batched_embeddings,
            indices,
            self.skip_empty,
        )

        _cached_empty_embedding: Optional[list[float]] = None

        def empty_embedding() -> list[float]:
            nonlocal _cached_empty_embedding
            if _cached_empty_embedding is None:
                average_embedded = self.client.create(input="",
                                                      extra_headers=self.headers,
                                                      **client_kwargs)
                if not isinstance(average_embedded, dict):
                    average_embedded = average_embedded.model_dump()
                _cached_empty_embedding = average_embedded["data"][0]["embedding"]
            return _cached_empty_embedding

        return [e if e is not None else empty_embedding() for e in embeddings]

    async def _aget_len_safe_embeddings(
            self,
            texts: list[str],
            *,
            engine: str,
            chunk_size: Optional[int] = None,
            **kwargs: Any,
    ) -> list[list[float]]:
        _chunk_size = chunk_size or self.chunk_size
        client_kwargs = {**self._invocation_params, **kwargs}

        token_chunks, decoded_chunks, indices = await run_in_executor(
            None, self._tiktoken_tokenize_with_decode, texts, chunk_size
        )

        batched_embeddings = await self._process_all_batches_embedding_async(decoded_chunks, client_kwargs)

        embeddings = _process_batched_chunked_embeddings(
            len(texts),
            token_chunks,
            batched_embeddings,
            indices,
            self.skip_empty,
        )

        _cached_empty_embedding: Optional[list[float]] = None

        async def empty_embedding() -> list[float]:
            nonlocal _cached_empty_embedding
            if _cached_empty_embedding is None:
                average_embedded = await self.async_client.create(input="",
                                                                  extra_headers=self.headers,
                                                                  **client_kwargs)
                if not isinstance(average_embedded, dict):
                    average_embedded = average_embedded.model_dump()
                _cached_empty_embedding = average_embedded["data"][0]["embedding"]
            return _cached_empty_embedding

        return [e if e is not None else await empty_embedding() for e in embeddings]


class DefaultEmbeddingsV2:

    @staticmethod
    def get_default_embedding_config() -> dict:
        for model in EMBEDDING_MODELS:
            if model['default']:
                return model
        return EMBEDDING_MODELS[0]

    @staticmethod
    def get_static_embeddings(
            is_chinese_text: bool = True,
            embedding_model_name: str = None,
            embedding_model_id: str = None,
            **kwargs
    ) -> Tuple[Embeddings, str]:
        if embedding_model_id is not None:
            embedding_config = next((model for model in EMBEDDING_MODELS if model['id'] == embedding_model_id), None)
            if embedding_config is None:
                raise ValueError(f"Unsupported embedding model id: {embedding_model_id}")
            embedding_model_name = embedding_config['model_name']
        elif embedding_model_name is not None and embedding_model_id is None:
            embedding_model_name = embedding_model_name
            embedding_config = next(
                (model for model in EMBEDDING_MODELS if model['model_name'] == embedding_model_name), None)
            if embedding_config is None:
                raise ValueError(f"Unsupported embedding model name: {embedding_model_name}")
        else:
            embedding_config = DefaultEmbeddingsV2.get_default_embedding_config()
            embedding_model_name = embedding_config['model_name']

        user_id = get_user_id()
        logger.info(f"[DefaultEmbeddingsV2]-get_static_embeddings- User: {user_id} using embedding model: {embedding_model_name} with config: {embedding_config}")
        embedding = OpenAICompatibleEmbeddings(
            user_id=user_id,
            openai_api_base=embedding_config['api_base'],
            model=embedding_model_name,
            embedding_model_id=embedding_config.get('id', None),
            batch_size=embedding_config.get('batch_size', 3),
            # Maximum number of batches to process concurrently
            sub_inputs_batch_size=embedding_config.get('sub_inputs_batch_size', 10),
            chunk_size=embedding_config.get('chunk_size', 2000),
            # Maximum number of texts to embed in each batch
            embedding_ctx_length=embedding_config.get('embedding_ctx_length', 8191),
            # The maximum number of tokens to embed at once.
            model_kwargs=embedding_config.get('model_kwargs', {}),
            api_key=embedding_config.get('api_key', None),
            max_retries=embedding_config.get('max_retries', None),
            dimensions=embedding_config.get('dimensions', None),
            headers=embedding_config.get('extra_header', None),
        )
        return embedding, embedding_model_name

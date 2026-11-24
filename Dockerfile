# 应用镜像：基于 base-nvidia-Dockerfile 产出的依赖镜像
#
# 构建依赖镜像：
#   docker build -t <your-registry>/lingro-rag-deps:latest -f base-nvidia-Dockerfile .
# 构建应用镜像：
#   docker build --build-arg BASE_IMAGE=<your-registry>/lingro-rag-deps:latest \
#                -t <your-registry>/lingro-rag-service:latest -f Dockerfile .
ARG BASE_IMAGE=lingro/lingro-rag-deps:latest
FROM ${BASE_IMAGE}

ENV LANG=C.UTF-8
# SCARF_NO_ANALYTICS=true 关闭 unstructured-io 的匿名数据采集
ENV SCARF_NO_ANALYTICS=true

VOLUME ["/tmp","/logs","/data","/models","/prompt_templates","/vector_store", "/make_kb_retryable_context_file_dir", "/bm25_index"]

# 复制项目代码到容器中
COPY --chown=root:root --chmod=755 . /app
WORKDIR /app

EXPOSE 80
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]

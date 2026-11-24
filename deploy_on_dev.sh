#!/usr/bin/env bash
# 开发环境：构建并推送 latest 镜像（可选打版本标签）
#
# 用法：
#   REGISTRY=registry.example.com/docker-local TAG_VERSION=v1.0.0 ./deploy_on_dev.sh
# 环境变量：
#   REGISTRY    必填，镜像仓库前缀（不含镜像名）
#   APP_NAME    可选，默认 lingro-rag-service
#   BASE_IMAGE  可选，依赖镜像地址，默认 $REGISTRY/$APP_NAME-deps:latest
set -euo pipefail

REGISTRY=${REGISTRY:?"set REGISTRY, e.g. REGISTRY=registry.example.com/docker-local"}
APP_NAME=${APP_NAME:-lingro-rag-service}
BASE_IMAGE=${BASE_IMAGE:-"$REGISTRY/$APP_NAME-deps:latest"}
TAG_VERSION=${1:-}

git pull --ff-only

docker build --build-arg BASE_IMAGE="$BASE_IMAGE" \
    -t "$REGISTRY/$APP_NAME:latest" -f Dockerfile .
docker push "$REGISTRY/$APP_NAME:latest"

if [ -n "$TAG_VERSION" ]; then
    docker tag "$REGISTRY/$APP_NAME:latest" "$REGISTRY/$APP_NAME:$TAG_VERSION"
    docker push "$REGISTRY/$APP_NAME:$TAG_VERSION"
fi

# 清理本地低于当前版本的旧镜像
if [ -n "$TAG_VERSION" ]; then
    for tag in $(docker images --format "{{.Repository}}:{{.Tag}}" \
        | grep "$REGISTRY/$APP_NAME:v" | grep -v "$TAG_VERSION" || true); do
        version=$(echo "$tag" | awk -F: '{print $NF}' | sed 's/v//')
        current=$(echo "$TAG_VERSION" | sed 's/v//')
        if [ "$(printf '%s\n' "$version" "$current" | sort -V | head -n1)" = "$version" ] \
            && [ "$version" != "$current" ]; then
            docker rmi "$tag"
            echo "Removed old image: $tag"
        fi
    done
fi

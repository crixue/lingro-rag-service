#!/usr/bin/env bash
# 生产环境：把已推送的 latest 镜像提升为正式版本号并推送
#
# 用法：
#   REGISTRY=registry.example.com/docker-release INITIAL_VERSION=v1.0.0 ./deploy_on_prod.sh
# 环境变量：
#   REGISTRY        必填，发布仓库前缀（不含镜像名）
#   APP_NAME        可选，默认 lingro-rag-service
#   INITIAL_VERSION 可选，不传则基于本地已有镜像的版本号自增
set -euo pipefail

REGISTRY=${REGISTRY:?"set REGISTRY, e.g. REGISTRY=registry.example.com/docker-release"}
APP_NAME=${APP_NAME:-lingro-rag-service}
INITIAL_VERSION=${1:-}

if [ -z "$INITIAL_VERSION" ]; then
    LATEST_VERSION=$(docker images --filter "reference=$REGISTRY/$APP_NAME" --format "{{.Tag}}" \
        | sort -V | tail -n 1)
    LATEST_VERSION=${LATEST_VERSION#v}
    INITIAL_VERSION=$(echo "$LATEST_VERSION" | awk -F. -v OFS=. \
        '{$3++; if ($3 >= 1000) {$3 = 0; $2++}; if ($2 >= 1000) {$2 = 0; $1++}; print}')
    INITIAL_VERSION="v$INITIAL_VERSION"
fi

docker tag "$REGISTRY/$APP_NAME:latest" "$REGISTRY/$APP_NAME:$INITIAL_VERSION"
docker push "$REGISTRY/$APP_NAME:$INITIAL_VERSION"

# 清理本地旧版本镜像
for tag in $(docker images --format "{{.Repository}}:{{.Tag}}" \
    | grep "$REGISTRY/$APP_NAME:v" | grep -v "$INITIAL_VERSION" || true); do
    version=$(echo "$tag" | awk -F: '{print $NF}' | sed 's/v//')
    current=$(echo "$INITIAL_VERSION" | sed 's/v//')
    if [ "$(printf '%s\n' "$version" "$current" | sort -V | head -n1)" = "$version" ] \
        && [ "$version" != "$current" ]; then
        docker rmi "$tag"
    fi
done

git tag -a "$INITIAL_VERSION" -m "Release version $INITIAL_VERSION"
git push origin "$INITIAL_VERSION"

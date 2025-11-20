#!/bin/bash
set -euo pipefail

echo "Life Cycle - BeforeInstall: Started."
echo "Installing Docker..."

apt-get update -y

# Docker 설치에 필요한 패키지 설치
apt-get install -y ca-certificates curl gnupg
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc

# Docker 리포지토리 추가
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  tee /etc/apt/sources.list.d/docker.list > /dev/null

# Docker 엔진, CLI, Compose 설치
apt-get update -y
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# ubuntu 사용자가 sudo 없이 docker 명령어 사용하도록 설정
usermod -aG docker ubuntu

echo "Life Cycle - BeforeInstall: Docker installation complete."

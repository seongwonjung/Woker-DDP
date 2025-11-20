#!/bin/bash

# 1. AWS Secrets Manager에서 .env 내용을 가져옵니다.
#    (EC2의 IAM 역할이 이 Secret에 대한 GetSecretValue 권한을 가지고 있어야 합니다.)
#    (EC2가 사용하는 리전으로 'ap-northeast-2'를 수정하세요)
#    ("my-app/env" 부분을 Secrets Manager에 저장한 실제 보안 암호 이름으로 변경하세요)
aws secretsmanager get-secret-value \
    --secret-id "worker/env" \
    --region ap-northeast-2 \
    --query SecretString \
    --output text | jq -r 'to_entries|map("\(.key)=\(.value)")|.[]' > /home/ubuntu/app/.env

# 2. 생성된 .env 파일의 소유자를 'ubuntu' 사용자로 변경합니다.
#    (appspec.yml의 destination 경로와 runas 사용자와 일치시킵니다.)
chown ubuntu:ubuntu /home/ubuntu/app/.env
chmod 600 /home/ubuntu/app/.env # (보안을 위해 소유자만 읽고 쓸 수 있도록 설정)

echo ".env file created successfully."

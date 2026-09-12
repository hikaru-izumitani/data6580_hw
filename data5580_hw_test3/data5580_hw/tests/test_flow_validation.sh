#!/bin/bash

BASE_URL="http://127.0.0.1:5000/users"

echo "=========================================="
echo "1. [検証] 不正なメールアドレス（@なし）でPOSTして400エラーを確認"
echo "=========================================="
curl -i -X POST "${BASE_URL}" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Invalid User 1",
    "email": "invalid-email-format"
  }'
echo -e "\n"

echo "=========================================="
echo "2. [検証] 不正なメールアドレス（ドメインなし）でPOSTして400エラーを確認"
echo "=========================================="
curl -i -X POST "${BASE_URL}" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Invalid User 2",
    "email": "test@"
  }'
echo -e "\n"

echo "=========================================="
echo "3. [検証] 正常なメールアドレスでPOSTして成功することを確認"
echo "=========================================="
curl -i -X POST "${BASE_URL}" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Valid User",
    "email": "valid_user@example.com"
  }'
echo -e "\n"
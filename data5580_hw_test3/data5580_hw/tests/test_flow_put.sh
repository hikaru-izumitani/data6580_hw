#!/bin/bash

BASE_URL="http://127.0.0.1:5000/users"

echo "=========================================="
echo "1. [準備] テスト用の初期ユーザーをPOSTで作成する"
echo "=========================================="
# 新規作成してレスポンスからIDを抽出（jqコマンドを使用）
RESPONSE=$(curl -s -X POST "${BASE_URL}" \
  -H "Content-Type: application/json" \
  -d '{"name": "Old Name", "email": "old_email@example.com"}')

echo "${RESPONSE}"
USER_ID=$(echo "${RESPONSE}" | jq -r '.id')
echo "生成された User ID: ${USER_ID}"
echo -e "\n"

echo "=========================================="
echo "2. [検証] PUTリクエストで全フィールドを更新する"
echo "=========================================="
# PUTでは name と email の両方を送信する
curl -i -X PUT "${BASE_URL}/${USER_ID}" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Updated Full Name",
    "email": "new_email@example.com"
  }'
echo -e "\n\n"

echo "=========================================="
echo "3. [確認] 更新されたユーザーをGETで取得して検証する"
echo "=========================================="
curl -s -X GET "${BASE_URL}/${USER_ID}" \
  -H "Content-Type: application/json"
echo -e "\n"
#!/bin/bash

BASE_URL="http://127.0.0.1:5000/users"

echo "=========================================="
echo "0. [準備] 削除テスト用の新規ユーザーをPOSTで作成する"
echo "=========================================="

RESPONSE=$(curl -s -X POST "${BASE_URL}" \
  -H "Content-Type: application/json" \
  -d '{"name": "Delete Target User", "email": "delete_test@example.com"}')

echo "Response: $RESPONSE"

USER_ID=$(echo "$RESPONSE" | jq -r '.id')
echo "取得したユーザーID: $USER_ID"
echo ""

if [ "$USER_ID" = "null" ] || [ -z "$USER_ID" ]; then
  echo "エラー: ユーザーIDの取得に失敗しました。"
  exit 1
fi

echo "=========================================="
echo "1. [正常系] ユーザーをDELETEで削除する"
echo "=========================================="
curl -i -X DELETE "${BASE_URL}/${USER_ID}"
echo -e "\n\n"

echo "=========================================="
echo "2. [確認] 削除されたユーザーにGETを送り、404が返ることを確認する"
echo "=========================================="
curl -i -X GET "${BASE_URL}/${USER_ID}"
echo -e "\n\n"

echo "=========================================="
echo "3. [異常系] 存在しないID（すこし適当なID）をDELETEして404を確認する"
echo "=========================================="
curl -i -X DELETE "${BASE_URL}/non_existent_id_99999"
echo -e "\n\n"
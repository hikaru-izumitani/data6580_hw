#!/bin/bash

BASE_URL="http://127.0.0.1:5000/users"

echo "=========================================="
echo "0. [準備] 新規ユーザーをPOSTで作成する"
echo "=========================================="

# curlのレスポンスをいったん変数に受け取る
# （※ 実際のPOSTのエンドポイントや送信データ構造に合わせて調整してください）
RESPONSE=$(curl -s -X POST "${BASE_URL}" \
  -H "Content-Type: application/json" \
  -d '{"name": "Test User", "email": "test_flow@example.com"}')

echo "Response: $RESPONSE"

# 返ってきたJSONから id を抽出する（jqを使用）
# ※ もし返り値のキーが 'id' ではなく別名であれば書き換えてください
USER_ID=$(echo "$RESPONSE" | jq -r '.id')

echo "取得したユーザーID: $USER_ID"
echo ""

# 万が一IDが取れなかった場合のガード
if [ "$USER_ID" == "null" ] || [ -z "$USER_ID" ]; then
  echo "エラー: ユーザーIDの取得に失敗しました。"
  exit 1
fi

echo "=========================================="
echo "1. [正常系] 今作ったユーザーの名前だけをPATCHで更新する"
echo "=========================================="
curl -i -X PATCH "${BASE_URL}/${USER_ID}" \
  -H "Content-Type: application/json" \
  -d '{"name": "Updated Test User"}'
echo -e "\n\n"

echo "=========================================="
echo "2. [正常系] 今作ったユーザーの情報をGETで確認する"
echo "=========================================="
curl -i -X GET "${BASE_URL}/${USER_ID}" \
  -H "Content-Type: application/json"
echo -e "\n\n"

echo "=========================================="
echo "3. [異常系] 空のJSONを送って400エラーを確認する"
echo "=========================================="
curl -i -X PATCH "${BASE_URL}/${USER_ID}" \
  -H "Content-Type: application/json" \
  -d '{}'
echo -e "\n\n"
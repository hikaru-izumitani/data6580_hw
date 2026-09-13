#!/bin/bash

BASE_URL="http://127.0.0.1:5000/users"

echo "=========================================="
echo "1. [準備] テストユーザー1をPOSTで作成する"
echo "=========================================="
curl -s -X POST "${BASE_URL}" \
  -H "Content-Type: application/json" \
  -d '{"name": "Alice Smith", "email": "alice_all@example.com"}'
echo -e "\n"

echo "=========================================="
echo "2. [準備] テストユーザー2をPOSTで作成する"
echo "=========================================="
curl -s -X POST "${BASE_URL}" \
  -H "Content-Type: application/json" \
  -d '{"name": "Bob Jones", "email": "bob_all@example.com"}'
echo -e "\n"

echo "=========================================="
echo "3. [検証] 全ユーザー一覧を取得する (GET /users)"
echo "=========================================="
curl -i -X GET "${BASE_URL}" \
  -H "Content-Type: application/json"
echo -e "\n\n"
from app import app, users
import pytest


@pytest.fixture
def client():
  app.config["TESTING"] = True
  # テストごとにユーザーリストを初期状態に戻す
  users.clear()
  users.extend([
      {"id": 1, "name": "Alice", "email": "alice@example.com"},
      {"id": 2, "name": "Bob", "email": "bob@example.com"},
  ])
  with app.test_client() as client:
    yield client


# --- 1, 5, 6, 9. POST (ユーザー作成 & バリデーション & 重複チェック) ---
def test_create_user_success(client):
  response = client.post(
      "/users", json={"name": "Charlie", "email": "charlie@example.com"}
  )
  assert response.status_code == 201
  data = response.get_json()
  assert data["user"]["id"] == 3
  assert data["user"]["name"] == "Charlie"


def test_create_user_empty_users(client):
  users.clear()
  response = client.post(
      "/users", json={"name": "Alice", "email": "alice@example.com"}
  )
  assert response.status_code == 201
  data = response.get_json()
  assert data["user"]["id"] == 1


def test_create_user_missing_fields(client):
  response = client.post("/users", json={"name": "Charlie"})
  assert response.status_code == 400
  assert "Missing required fields" in response.get_json()["error"]


def test_create_user_invalid_email(client):
  response = client.post(
      "/users", json={"name": "Charlie", "email": "invalid-email"}
  )
  assert response.status_code == 400
  assert response.get_json()["error"] == "Invalid email format"


def test_create_user_duplicate_email(client):
  response = client.post(
      "/users", json={"name": "Charlie", "email": "alice@example.com"}
  )
  assert response.status_code == 400
  assert response.get_json()["error"] == "Email is already in use"


# --- 5. GET (全ユーザー取得) ---
def test_get_all_users(client):
  response = client.get("/users")
  assert response.status_code == 200
  data = response.get_json()
  assert len(data) == 2


# --- 2. GET (特定ユーザー取得) ---
def test_get_user_success(client):
  response = client.get("/users/1")
  assert response.status_code == 200
  assert response.get_json()["user"]["name"] == "Alice"


def test_get_user_not_found(client):
  response = client.get("/users/999")
  assert response.status_code == 404
  assert response.get_json()["error"] == "The user was not found"


# --- 3, 8. PATCH (部分更新) ---
def test_update_patch_name_only(client):
  response = client.patch("/users/1", json={"name": "Alice Updated"})
  assert response.status_code == 200
  assert response.get_json()["user"]["name"] == "Alice Updated"
  assert response.get_json()["user"]["email"] == "alice@example.com"


def test_update_patch_email_only(client):
  response = client.patch("/users/1", json={"email": "newalice@example.com"})
  assert response.status_code == 200
  assert response.get_json()["user"]["email"] == "newalice@example.com"


def test_update_patch_missing_body(client):
  response = client.patch("/users/1", json={})
  # ボディが空か、フィールドが全くないケース
  assert response.status_code == 400


def test_update_patch_no_fields(client):
  response = client.patch("/users/1", json={"unknown": "value"})
  assert response.status_code == 400


def test_update_patch_invalid_email(client):
  response = client.patch("/users/1", json={"email": "bad-email"})
  assert response.status_code == 400


def test_update_patch_duplicate_email(client):
  # BobのメールアドレスにAliceを変更しようとする
  response = client.patch("/users/1", json={"email": "bob@example.com"})
  assert response.status_code == 400


def test_update_patch_not_found(client):
  response = client.patch("/users/999", json={"name": "Nobody"})
  assert response.status_code == 404


# --- 7, 8. PUT (全置換更新) ---
def test_update_put_success(client):
  response = client.put(
      "/users/1", json={"name": "Alice Put", "email": "alice.put@example.com"}
  )
  assert response.status_code == 200
  assert response.get_json()["user"]["name"] == "Alice Put"


def test_update_put_missing_body(client):
  response = client.put("/users/1", json=None)
  assert response.status_code == 400


def test_update_put_missing_fields(client):
  response = client.put("/users/1", json={"name": "Alice Only"})
  assert response.status_code == 400
  assert "Missing required fields" in response.get_json()["error"]


def test_update_put_invalid_email(client):
  response = client.put("/users/1", json={"name": "Alice", "email": "bad"})
  assert response.status_code == 400


def test_update_put_duplicate_email(client):
  response = client.put(
      "/users/1", json={"name": "Alice", "email": "bob@example.com"}
  )
  assert response.status_code == 400


def test_update_put_not_found(client):
  response = client.put(
      "/users/999", json={"name": "Ghost", "email": "ghost@example.com"}
  )
  assert response.status_code == 404


# --- 4. DELETE (ユーザー削除) ---
def test_delete_user_success(client):
  response = client.delete("/users/1")
  assert response.status_code == 200
  assert response.get_json()["message"] == "User deleted successfully"


def test_delete_user_not_found(client):
  response = client.delete("/users/999")
  assert response.status_code == 404
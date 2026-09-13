import uuid
import pytest

from data5580_hw.app import create_app

@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True

    with app.test_client() as client:
        yield client


def test_home_route(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json == {"message": "Hello World!"}

#Hikaru Izumitani
def test_create_user(client):
    """test for Acceptance Criteria 1"""
    """AC 1-1"""
    # Acceptance Criteria 1-1: The system should accept a POST request at /users with required user details (e.g., name, email).

    payload = {
        "name": "Hikaru"
        ,"email": "hikaru@example.com"
    }
    """AC 1-3"""
    # Acceptance Criteria 1-3: If the creation is successful, return a success message with the user’s ID and details.
    response = client.post("/users", json=payload)
    #print("DEBUG RESPONSE:", response.json)
    assert response.json["name"] == "Hikaru"
    assert response.json["email"] == "hikaru@example.com"
    assert response.status_code ==200
    assert response.json != {}
    assert "id" in response.json
    assert response.json["id"] is not None
    """AC 1-2"""
    # Acceptance Criteria 1-2: If the user already exists, return an error with a message indicating duplication.
    response2 = client.post("/users", json=payload)
    assert response2.status_code == 400

    data = response2.get_json()
    assert "error" in data or "the email is already in use" in data
    "AC 1-4"
    user1_id = response.json["id"]
    payload3 = {
        "name": "Zack"
        , "email": "zack@example.com"
    }
    response3 = client.post("/users", json=payload3)
    user3_id = response3.json["id"]

    assert user1_id is not None
    assert user3_id is not None
    assert user1_id != user3_id

def test_get_user_by_id(client):
    """Acceptance Criteria 2"""
    """Setting"""
    unique_email = f"detail_{uuid.uuid4().hex[:8]}@example.com"
    post_res = client.post("/users", json={"name": "DetailUser", "email": unique_email})
    user_id = post_res.json["id"]
    """AC 2-1"""
    response = client.get(f"/users/{user_id}")
    assert response.status_code ==200
    """AC 2-2"""
    assert response.json["id"] == user_id
    assert response.json["name"] == "DetailUser"
    assert response.json["email"] == unique_email
    """AC 2-3"""
    fake_id = "42fewghbfte"
    response2 = client.get(f"/users/{fake_id}")
    assert response2.status_code==404

def test_path_user_by_id(client):
    """Acceptance Criteria 3"""
    """Setting"""
    unique_email = f"hikaru_old@example.com"
    post_res = client.post("/users", json={"name": "hikaru_old", "email": unique_email})
    user_id = post_res.json["id"]
    """AC 3-1"""
    payload = {
        "name": "Hikaru_new"
        ,"email":"Hikaru_new@example.com"
    }
    response = client.patch(f"/users/{user_id}", json=payload)
    assert response.status_code == 200

    payload2 = {
        "name": "Takumi_new"
    }
    response2 = client.patch(f"/users/{user_id}", json=payload2)
    assert response2.status_code == 200

    payload3 = {
        "email": "takumi_new@example.com"
    }
    response3 = client.patch(f"/users/{user_id}", json = payload3)
    assert response3.status_code == 200

    fake_user_id = "42fewghbfte"
    payload4 = {
        "name": "john_fake"
        ,"email": "john_fake@example.com"
    }
    response_fake = client.patch(f"/users/{fake_user_id}", json = payload4)
    assert response_fake.status_code == 404




"""
def test_add_success(client):
    response = client.post("/add", json={"a": 2, "b": 3})
    assert response.status_code == 200
    assert response.json["result"] == 5


def test_add_missing_values(client):
    response = client.post("/add", json={"a": 2})
    assert response.status_code == 400
    assert "error" in response.json
"""
import uuid
import pytest

from data5580_hw.app import create_app
INVALID_EMAIL_PAYLOAD = {
    "name": "Hikaru",
    "email": "rjfioefjrioeexamplecom"
}

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
def test_fetch_all_users(client):
    """Acceptance Criteria 5"""
    get_res = client.get("/users")
    assert get_res.status_code == 404
    assert get_res.json["error"] == "User not found"

    payload1 = {
        "name": "User1"
        ,"email": "User1@example.com"
        }
    response1 = client.post("/users", json=payload1)
    assert response1.status_code == 200
    payload2 = {
            "name": "User2"
            ,"email": "User2@example.com"
            }
    response2 = client.post("/users", json=payload2)
    assert response2.status_code == 200
    get_res2 = client.get("/users")
    assert get_res2.status_code == 200
    
    users = get_res2.json
    assert isinstance(users, list) 
    assert len(users) >= 2 
    
    user_names = [u["name"] for u in users]
    assert "User1" in user_names
    assert "User2" in user_names

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
    assert response.json["message"] == "User created successfully."
    assert response.status_code ==200
    assert response.json != {}
    assert "id" in response.json
    assert response.json["id"] is not None
    """AC 1-2"""
    # Acceptance Criteria 1-2: If the user already exists, return an error with a message indicating duplication.
    response2 = client.post("/users", json=payload)
    """Acceptance Criteria 6"""
    assert response2.json["error"] == "the email is already in use"
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
    """Acceptance Criteria 9"""
    response_email_bug = client.post("/users", json=INVALID_EMAIL_PAYLOAD)
    assert response_email_bug.json["error"] == "Invalid email format"
    assert response_email_bug.status_code == 400

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

def test_patch_user_by_id(client):
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
    assert response.json["id"] == user_id
    assert response.json["name"] == "Hikaru_new"
    assert response.json["email"] == "Hikaru_new@example.com"
    assert response.json["message"] == "User updated successfully."
    assert response.status_code == 200

    payload2 = {
        "name": "Takumi_new"
    }
    response2 = client.patch(f"/users/{user_id}", json=payload2)
    #assert response2.json["id"] == user_id
    assert response2.json["name"] == "Takumi_new"
    #assert response2.json["email"] == "Hikaru_new@example.com"
    #assert response2.json["message"] == "User updated successfully."
    assert response2.status_code == 200

    payload3 = {
        "email": "takumi_new@example.com"
    }
    response3 = client.patch(f"/users/{user_id}", json = payload3)

    assert response3.status_code == 200
    #assert response3.json["id"] == user_id
    #assert response3.json["name"] == "Takumi_new"
    assert response3.json["email"] == "takumi_new@example.com"
    #assert response3.json["message"] == "User updated successfully."
    assert response3.status_code == 200

    """Acceptance Criteria 8"""
    fake_user_id = "42fewghbfte"
    payload4 = {
        "name": "john_fake"
        ,"email": "john_fake@example.com"
    }
    response_fake = client.patch(f"/users/{fake_user_id}", json = payload4)
    assert response_fake.status_code == 404

    response4 = client.patch(f"/users/{user_id}", json={})
    assert response4.json["error"] == "No valid fields to update. You need name or email to update"
    assert response4.status_code == 400

    """Acceptance Criteria 9"""
    response_email_bug = client.patch(f"/users/{user_id}", json=INVALID_EMAIL_PAYLOAD)
    assert response_email_bug.json["error"] == "Invalid email format"
    assert response_email_bug.status_code == 400

def test_delete_user_by_id(client):
    """Acceptance Criteria 4"""
    """Setting"""
    unique_email = f"detail_{uuid.uuid4().hex[:8]}@example.com"
    post_res = client.post("/users", json={"name": "User_for_delete", "email": unique_email})
    user_id = post_res.json["id"]
    response = client.get(f"/users/{user_id}")
    assert response.status_code ==200
    assert response.json["id"] == user_id
    assert response.json["name"] == "User_for_delete"
    assert response.json["email"] == unique_email

    response2 = client.delete(f"/users/{user_id}")
    assert response2.status_code == 200
    assert response2.json["message"] == "User deleted successfully."

    fake_id = "fakefakefake"
    response_fake = client.delete(f"/users/{fake_id}")
    assert response_fake.status_code == 404
    assert response_fake.json["error"] == "User not found"

def test_put_user_by_id(client):
    """Acceptance Criteria 7"""
    """Setting"""
    unique_email = f"ben_old@example.com"
    post_res = client.post("/users", json={"name": "Ben_old", "email": unique_email})
    user_id = post_res.json["id"]
    payload = {
        "name": "Ben_new"
        ,"email":"Ben_new@example.com"
    }
    response = client.put(f"/users/{user_id}", json=payload)
    assert response.json["id"] == user_id
    assert response.json["name"] == "Ben_new"
    assert response.json["email"] == "Ben_new@example.com"
    assert response.json["message"] == "User updated successfully."
    assert response.status_code == 200
    """Acceptance Criteria 8"""
    payload2 = {
        "name": "Alice_new"
    }
    response2 = client.put(f"/users/{user_id}", json=payload2)
    assert response2.json["error"] == "You need email to update."
    assert response2.status_code == 400

    payload3 = {
        "email": "Alice_new@example.com"
    }
    response3 = client.put(f"/users/{user_id}", json = payload3)
    assert response3.json["error"] == "You need name to update."
    assert response3.status_code== 400

    fake_user_id = "481grwfwe34"
    payload4 = {
        "name": "john_fake5"
        ,"email": "john_fake5@example.com"
    }
    response_fake = client.put(f"/users/{fake_user_id}", json = payload4)
    assert response_fake.json["error"] == "User not found"
    assert response_fake.status_code == 404
    """Acceptance Criteria 9"""
    response_email_bug = client.put(f"/users/{user_id}", json=INVALID_EMAIL_PAYLOAD)
    assert response_email_bug.json["error"] == "Invalid email format"
    assert response_email_bug.status_code == 400

    response_empty = client.put(f"/users/{user_id}", json={})
    assert response_empty.json["error"] == "No valid fields to update. You need name and email."

def test_create_user_invalid_email_type(client):    
    response_empty = client.post("/users", json={
        "name": "Hikaru",
        "email": ""
    })
    assert response_empty.status_code == 400
    assert response_empty.json["error"] == "Invalid email format"  

    response_number = client.post("/users", json={
        "name": "Hikaru",
        "email": 12345678  # 数値を渡す
    })
    assert response_number.status_code == 400

def test_create_user_missing_keys(client):
    response = client.post("/users", json={
        "name": "Hikaru"
        # no email
    })
    assert response.status_code == 400
    assert "error" in response.json
def test_patch_put_user_duplicate_email(client):
    
    client.post("/users", json={
        "name": "UserOne",
        "email": "existing@example.com"
    })

    res2 = client.post("/users", json={
        "name": "UserTwo",
        "email": "twouser@example.com"
    })
    user2_id = res2.json["id"]

    patch_response = client.patch(f"/users/{user2_id}", json={
        "email": "existing@example.com"
    })

    assert patch_response.status_code == 400
    assert patch_response.json["error"] == "the email is already in use"

    put_response = client.put(f"/users/{user2_id}", json ={
        "name": "new_existing"
        ,"email": "existing@example.com"
    })
    assert put_response.status_code == 400
    assert put_response.json["error"] == "the email is already in use"

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
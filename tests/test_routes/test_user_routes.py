import pytest

from data5580_hw.routes.user_routes import user_controller

from tests.conftest import client


def test_create_user(client, mocker):
    mocker.patch.object(
        user_controller,
        "create_user",
        return_value=({"id": "1", "name": "admin"}, 200)
    )

    response = client.post(
        "/user",
        json={
            "name": "admin",
            "email": "test@test.com"
        }
    )

    assert response.status_code == 200
    assert response.json["name"] == "admin"


def test_create_user_missing_name(client, mocker):
    mocker.patch.object(
        user_controller,
        "create_user",
        return_value=({"error": "Missing name"}, 400)
    )

    response = client.post(
        "/user",
        json={
            "email": "test@test.com"
        }
    )

    assert response.status_code == 400


def test_create_user_missing_email(client, mocker):
    mocker.patch.object(
        user_controller,
        "create_user",
        return_value=({"error": "Missing email"}, 400)
    )

    response = client.post(
        "/user",
        json={
            "name": "admin"
        }
    )

    assert response.status_code == 400


def test_get_user(client, mocker):
    mocker.patch.object(
        user_controller,
        "get_user",
        return_value=({
            "id": "1",
            "name": "admin",
            "email": "test@test.com"
        }, 200)
    )

    response = client.get("/user/1")

    assert response.status_code == 200
    assert response.json["id"] == "1"


def test_get_user_not_found(client, mocker):
    mocker.patch.object(
        user_controller,
        "get_user",
        return_value=({"error": "User not found"}, 404)
    )

    response = client.get("/user/999")

    assert response.status_code == 404
    assert response.json["error"] == "User not found"

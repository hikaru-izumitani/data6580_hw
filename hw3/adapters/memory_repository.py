from ports.repository import UserRepositoryPort

class InMemoryUserRepository(UserRepositoryPort):
    def __init__(self):
        self.users = [
            {"id": 1, "name": "Alice", "email": "alice@example.com"},
            {"id": 2, "name": "Bob", "email": "bob@example.com"}
        ]
    def get_all(self):
        return self.users

    def get_by_id(self, user_id):
        for user in self.users:
            if user["id"] == user_id:
                return user
        return None
    

    def find_by_email(self, email):
        for user in self.users:
            if user["email"] == email:
                return user
        return None

    def save(self, name: str, email: str):
            next_id = max(user["id"] for user in self.users) + 1 if self.users else 1
            new_user = {
                "id": next_id,
                "name": name,
                "email": email
            }
            self.users.append(new_user)
            return new_user
    def update(self, user_id: int, name: str = None, email: str = None):
        user = self.get_by_id(user_id)
        if not user:
            return None
        if name is not None:
            user["name"] = name
        if email is not None:
            user["email"] = email
        return user

    def delete(self, user_id: int):
        user = self.get_by_id(user_id)
        if user:
            self.users.remove(user)
            return True
        return False
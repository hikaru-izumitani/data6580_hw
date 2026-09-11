import re
from ports.repository import UserRepositoryPort
# REGEX Reference: WHATWG HTML Living Standard (E-mail state)
# https://html.spec.whatwg.org/multipage/input.html#email-state-(type=email)
EMAIL_REGEX = r"^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*$"

class UserService:
    def __init__(self, repo: UserRepositoryPort):
        self.repo = repo

    def _validate_email(self, email, current_user_id=None):
        if not email or not re.match(EMAIL_REGEX, email):
            return "Invalid email format"

        all_users = self.repo.get_all()
        for user in all_users:
            if current_user_id is not None and user["id"] == current_user_id:
                continue
            if user["email"] == email:
                return "Email is already in use"
        return None

    def list_users(self):
        return self.repo.get_all()

    def get_user(self, user_id: int):
        return self.repo.get_by_id(user_id)

    def create_user(self, name: str, email: str):
        email_error = self._validate_email(email)
        if email_error:
            return None, email_error
        
        new_user = self.repo.save(name, email)
        return new_user, None

    def update_user(self, user_id: int, name: str = None, email: str = None):
        if email:
            email_error = self._validate_email(email, current_user_id=user_id)
            if email_error:
                return None, email_error

        updated_user = self.repo.update(user_id, name, email)
        return updated_user, None

    def delete_user(self, user_id: int):
        return self.repo.delete(user_id)
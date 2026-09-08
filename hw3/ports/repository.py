from abc import ABC, abstractmethod

class UserRepository(ABC):
    @abstractmethod
    def get_all(self):
        pass
    @abstractmethod
    def get_by_id(self, user_id: int):
        pass
    @abstractmethod
    def save(self, name: str, email: str):
        pass
    @abstractmethod
    def update(self, user_id: int, name: str = None, email: str = None):
        pass
    @abstractmethod
    def delete(self, user_id: int):
        pass
    @abstractmethod
    def find_by_email(self, email: str):
        pass
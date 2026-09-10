from datetime import datetime
from models.user_model import UserSQL
from services.database.database_client import db

class SQLAlchemyUserRepository:
    """A repository class for managing user data using SQLAlchemy."""

    def create(self, user_id: str, name: str, email: str) -> UserSQL:
        """save a new user to the database"""
        #now = datetime.utcnow()
        user = UserSQL(
            id=user_id,
            name=name,
            email=email#,
            #created=now,
            #updated=now
        )
        db.session.add(user)
        db.session.commit()
        return user

    def find_by_id(self, user_id: str) -> UserSQL | None:
        """find a user by ID"""
        return db.session.get(UserSQL, user_id)

    def find_all(self) -> list[UserSQL]:
        """get all users"""
        stmt = db.select(UserSQL)
        return list(db.session.scalars(stmt).all())

    def update(self, user_id: str, name: str = None, email: str = None) -> UserSQL | None:
        """update a user by ID"""
        user = self.find_by_id(user_id)
        if not user:
            return None

        if name is not None:
            user.name = name
        if email is not None:
            user.email = email
        
        user.updated = datetime.utcnow()
        db.session.commit()
        return user

    def delete(self, user_id: str) -> bool:
        """delete a user by ID"""
        user = self.find_by_id(user_id)
        if not user:
            return False

        db.session.delete(user)
        db.session.commit()
        return True
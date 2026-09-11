# File 6 user_sql.py for File 4 gateways/user_repository.py
from sqlalchemy import UniqueConstraint
from data5580_hw.services.database.database_client import db

class UserSQL(db.Model):
    __tablename__="users"
    id = db.Column(db.String, primary_key=True)
    name = db.Column(db.String)
    email = db.Column(db.String)
    created = db.Column(db.DateTime)
    updated = db.Column(db.DateTime)

    __table_args__ = (UniqueConstraint((email), name="uq_user_email" ),)

    def to_dict(self):
        return {"id": self.id, "name": self.name, "email": self.email}
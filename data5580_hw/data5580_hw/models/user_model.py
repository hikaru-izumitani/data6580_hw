# models/user_model.py の例
from services.database.database_client import db  # または相対パス from ..gateways.database_client import db


class UserSQL(db.Model):  # ← 必ず db.Model を継承していること
  __tablename__ = "users"

  id = db.Column(db.String, primary_key=True)
  name = db.Column(db.String, nullable=False)
  email = db.Column(db.String, nullable=False)

  def to_dict(self):
    return {"id": self.id, "name": self.name, "email": self.email}

"""from dataclasses import dataclass
from datetime import datetime

@dataclass
class UserSQL(object):
    id: str
    name: str
    email: str
    created: datetime
    updated: datetime
"""
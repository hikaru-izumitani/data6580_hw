from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

def init_db(app) -> None:
   import data5580_hw.services.database.user_sql

   with app.app_context():
       db.create_all()

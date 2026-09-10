def setup_database(app):
    from data5580_hw.services.database.database_client import db
    db.init_app(app)
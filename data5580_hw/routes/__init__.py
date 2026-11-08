

def init_blueprints(app) -> None:

    from data5580_hw.routes.user_routes import user_blueprint
    app.register_blueprint(user_blueprint)

    from data5580_hw.routes.prediction_routes import prediction_blueprint
    app.register_blueprint(prediction_blueprint)

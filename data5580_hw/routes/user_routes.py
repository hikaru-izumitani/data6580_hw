from flask import Blueprint

# In-app modules
from data5580_hw.controllers.user_controller import user_controller

user_blueprint = Blueprint('user', __name__)


@user_blueprint.route('/create', methods=['POST'])
def create_user():
    return user_controller.create_user()

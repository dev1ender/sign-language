from flask import render_template, request, redirect, Blueprint
import uuid
from auth.utils import isLogged

auth_blueprint = Blueprint('login_page', __name__,
                       template_folder='templates')

login_config = {
    'username': 'admin',
    'password': 'admin@2023',
}


def generate_token():
    rand_token = uuid.uuid4().hex
    return rand_token


@auth_blueprint.route("/login", methods=['POST', 'GET'])
def login():
    redirect_res = redirect('/')

    # redirect home if is logged
    if isLogged():
        return redirect_res

    if request.method == 'POST':
        user = request.form['user']
        password = request.form['pass']
        if user == login_config["username"] and password == login_config['password']:
            redirect_res.set_cookie('AuthToken', generate_token())
        return redirect_res
    return render_template("login.html")


@auth_blueprint.route("/logout", methods=['GET'])
def logout():
    res = redirect('/login')
    res.delete_cookie('AuthToken')
    return res

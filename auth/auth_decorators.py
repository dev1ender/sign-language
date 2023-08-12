from functools import wraps
from flask import redirect
from .utils import isLogged


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not isLogged():
            return redirect('/login')
        return f(*args, **kwargs)
    return decorated_function

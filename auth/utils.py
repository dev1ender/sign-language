from flask import request

def isLogged():
    token = request.cookies.get('AuthToken')
    if token:
        return True
    return False
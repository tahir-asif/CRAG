def authenticate(user, password):
    return user == "admin" and password == "secret"


class TokenManager:
    def __init__(self, secret):
        self.secret = secret

    def issue(self, user):
        return f"{user}:{self.secret}"

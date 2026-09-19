def parse_token(token):
    user, _, secret = token.partition(":")
    return user, secret

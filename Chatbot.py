import secretstorage

class Chatbot:
    # Setup secretstorage:
    _bus = secretstorage.dbus_init()
    _collection = secretstorage.get_default_collection(_bus)

    def __init__(self, name):
        self.name = name

    def retrieve_key(self, api_name):
        key_search = self._collection.search_items({'api': api_name})
        key = next(key_search)

        if key:
            secret = key.get_secret().decode('utf-8')
            return secret
        else:
            return False

    def send_message(self, user_message):
        raise NotImplementedError

        # Send dialogue history to chatbot:
        # response_message = self.api.send_request(user_message)
        # return response_message

    def request_tokens(self):
        raise NotImplementedError

        # Request tokens from chatbot:
        # response_message = self.api.request_tokens()
        # return response_message

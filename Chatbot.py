import secretstorage

class Chatbot:
    """
    A chatbot that can be used to send messages to and receive messages from
    a chatbot API.
    """

    # Setup secretstorage:
    _bus = secretstorage.dbus_init()
    _collection = secretstorage.get_default_collection(_bus)

    # context: str = ""

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

    def send_message(self,
                     message,
                     stop_sequences = [],
                     #@REVISIT regex on every token is probably a bad idea, usually
                     stop_regex = None):
        raise NotImplementedError
        # response_message = self.api.send_request(message)
        # return response_message

    def request_tokens(self):
        raise NotImplementedError
        # response_message = self.api.request_tokens()
        # return response_message

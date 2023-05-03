import keyring
import appdirs

class Chatbot:
    """
    A chatbot that can be used to send messages to and receive messages from
    a chatbot API.
    """

    name: str
    keeps_context: bool = False
    # context: str = ""

    logdir: str = ""

    def __init__(self, name = "Chatbot", logdir = ""):
        self.name = name

        if logdir:
            self.logdir = logdir
        else:
            self.logdir = appdirs.user_log_dir('cammy', 'chatbots')

    def retrieve_key(self, api_name):
        key = keyring.get_password('api', api_name)

        if key:
            return key
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

    def log(self, message, filename = None):
        if not filename:
            filename = self.name + "-log.txt"

        # Log message to file
        with open(f"{self.logdir}/{filename}", 'a') as logfile:
            logfile.write(message + "\n")

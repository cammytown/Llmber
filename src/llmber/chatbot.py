import keyring
import appdirs

class Chatbot:
    """
    A chatbot that can be used to send messages to and receive messages from
    a chatbot API.
    """

    model_config: dict
    valid_options = []

    name: str
    keep_context: bool = False
    keep_response_in_context: bool = True
    # context: str = ""

    logdir: str = ""

    def __init__(self,
                 name: str = "Chatbot",
                 model_config: dict = {},
                 logdir: str = ""):
        self.name = name

        if logdir != "":
            self.logdir = logdir
        else:
            self.logdir = appdirs.user_log_dir('llmber', 'llmber')

        # Parse model_config
        self.validate_model_config(model_config)
        options = [ "keep_context", "keep_response_in_context" ]
        for option, value in model_config.items():
            if option in options:
                setattr(self, option, value) #@REVISIT
        self.model_config = model_config

    def retrieve_key(self, api_name):
        key = keyring.get_password('api', api_name)

        if key:
            return key
        else:
            return False

    def validate_model_config(self, model_config):
        """
        Validate model_config against valid_options.
        """
        for option in model_config.keys():
            if option not in self.valid_options:
                raise ValueError(f"Invalid option for selected model: {option}")

    def send_message(self,
                     message,
                     stop_sequences = [],
                     stop_regex = None,
                     n_tokens = 128,
                     ):
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
        with open(f"{self.logdir}/{filename}", 'a+') as logfile:
            logfile.write(message + "\n")

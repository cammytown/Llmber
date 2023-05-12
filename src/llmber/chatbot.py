import sys
import re
import appdirs
import keyring

class Chatbot:
    """
    A chatbot that can be used to send messages to and receive messages from
    a chatbot API.
    """

    model_config: dict
    valid_options = []

    name: str

    saved_contexts: list = []
    bos_token: int
    eos_token: int

    is_remote: bool = False
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

    def tokenize(self, message):
        raise NotImplementedError

    def detokenize(self, tokens):
        raise NotImplementedError

    def add_tokens_to_context(self, tokens):
        raise NotImplementedError

    def get_context(self):
        raise NotImplementedError

    def set_context(self, context):
        raise NotImplementedError

    def save_context(self):
        self.saved_contexts.append(self.get_context())

    def restore_context(self):
        if len(self.saved_contexts) > 0:
            self.set_context(self.saved_contexts.pop())
        else:
            print(f"WARN: No saved contexts to restore", file=sys.stderr)

    def sample(self,
               temp = 0.8,
               top_k = 30,
               top_p = 0.95,
               repeat_penalty = 1.3):
        raise NotImplementedError

    def send_message(self,
                     message,
                     stop_sequences = [],
                     stop_regex = None,
                     n_tokens = 128,
                     ):
        # Tokenize message
        if message == "":
            message = " " #@SCAFFOLDING

        inputs = self.tokenize(message)

        # Add tokens to context
        self.add_tokens_to_context(inputs)

        if __debug__:
            print(message, flush=True, end="")

        # Save context if necessary
        if not self.keep_response_in_context:
            self.save_context()

        # Generate response
        response_tokens = self.request_tokens(n_tokens=n_tokens,
                                              stop_sequences=stop_sequences)

        # Restore context if necessary
        if not self.keep_response_in_context:
            self.restore_context()

        # Decode the generated response
        response_text = self.detokenize(response_tokens)

        # if __debug__:
        #     print("Response:", response_text)

        return response_text

        # raise NotImplementedError
        # response_message = self.api.send_request(message)
        # return response_message

    def request_tokens(self, n_tokens = 128, stop_sequences = []):
        # Generate one token at a time
        response_tokens = []
        response_text = "" #@REVISIT optimization? only use if regex is needed?

        # Parse stop_sequences into a dictionary of filter types
        stop_filters = self.parse_stop_sequences(stop_sequences)

        for i in range(n_tokens):
            next_token = self.sample(temp=0.8,
                                     top_k=30,
                                     top_p=0.95,
                                     repeat_penalty=1.3)

            # Check if token is beginning-of-sequence token
            if self.bos_token and next_token == self.bos_token:
                continue

            # Check if the token is end-of-sequence token
            if self.eos_token and next_token == self.eos_token:
                break

            # Add token to model's context
            self.add_tokens_to_context([next_token])

            # Add token to response
            response_tokens.append(next_token)

            # Decode the token
            token_meaning = self.detokenize([next_token])

            # Add token meaning to response text
            response_text += token_meaning

            # Check for occurrences of stop sequences
            if self.check_stop_filters(stop_filters,
                                       response_tokens,
                                       response_text):
                break

            # Print the token
            if __debug__:
                print(token_meaning, flush=True, end="")

        if __debug__:
            print("", flush=True)
            sys.stdout.flush()

        return response_tokens

    def parse_stop_sequences(self, stop_sequences):
        """
        Parse a list of stop sequences into a dictionary of filter types.
        """

        stop_filters = {
            "token_sequences": [],
            "regexes": []
        }

        for stop_sequence in stop_sequences:
            # If stop sequence is a string
            if isinstance(stop_sequence, str):
                stop_tokens = self.tokenize(stop_sequence)
                stop_filters["token_sequences"].append(stop_tokens)

            # If stop sequence is a dict
            elif isinstance(stop_sequence, dict):
                match stop_sequence["type"]:
                    case "regex":
                        stop_filters["regexes"].append(stop_sequence["value"])

        return stop_filters

    def check_stop_filters(self, stop_filters, response_tokens, response_text):
        # Check for stop sequences that are token sequences
        for stop_token_seq in stop_filters["token_sequences"]:
            if len(response_tokens) >= len(stop_token_seq):
                if response_tokens[-len(stop_token_seq):] == stop_token_seq:
                    return True

        # Check for stop sequences that are regexes
        for stop_regex in stop_filters["regexes"]:
            if re.search(stop_regex, response_text):
                return True

        return False

    def log(self, message, filename = None):
        if not filename:
            filename = self.name + "-log.txt"

        # Log message to file
        with open(f"{self.logdir}/{filename}", 'a+') as logfile:
            logfile.write(message + "\n")

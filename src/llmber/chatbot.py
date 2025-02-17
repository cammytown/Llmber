import sys
import os
import re
import appdirs
import keyring
from typing import Optional

class Chatbot:
    """
    A chatbot that can be used to send messages to and receive messages from
    a chatbot API.
    """

    model_config: dict
    valid_options = []

    name: str

    saved_states: list = []

    temperature: float = 0.8
    top_k: int = 30
    top_p: float = 0.95
    repeat_penalty: float = 1.1
    presence_penalty: float = 0.0

    bos_token: Optional[int] = None
    eos_token: Optional[int] = None
    max_context_length: Optional[int] = None

    #@REVISIT not always in use
    token_occurrence_count: dict = {}

    context_string: str = "" #@REVISIT not always in use
    is_remote: bool = False
    keep_context: bool = False
    keep_response_in_context: bool = True

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
        """
        """

        # If api_env_var is set in model_config
        #@REVISIT feel like this is ugly architecture
        if "api_env_var" in self.model_config:
            # Retrieve key from environment variable
            key = os.getenv(self.model_config["api_env_var"])

        # Otherwise, retrieve key from keyring
        else:
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

    def add_string_to_context(self, string):
        if string == "":
            return

        # Tokenize the string
        tokens = self.tokenize(string)

        # Add tokens to context
        self.add_tokens_to_context(tokens)

        # Add string to context string
        self.context_string += string

    def get_state(self):
        """
        Get the current state of the chatbot.

        This is the internal state of the chatbot. When set_state is used
        with the value that get_state returns, the chatbot should be in the
        same state as it was when get_state was called.

        The state can be, for example, the logits and past_key_values of a
        HuggingFace Transformers model. Otherwise, it could simply be a
        string containing the chatbot's current context as in the case of
        remote chatbots like OpenAI.
        """

        raise NotImplementedError

    def set_state(self, state):
        """
        Set the current state of the chatbot.

        This is the internal state of the chatbot. When set_state is used
        with the value that get_state returns, the chatbot should be in the
        same state as it was when get_state was called.
        """

        raise NotImplementedError

    def save_state(self):
        """
        Save the current state of the chatbot.
        """

        self.saved_states.append(self.get_state())

    def restore_state(self):
        """
        Restore the most recently saved state of the chatbot.
        """

        if len(self.saved_states) > 0:
            self.set_state(self.saved_states.pop())
        else:
            print(f"WARN: No saved states to restore", file=sys.stderr)

    def clear_state(self):
        """
        Clear the current state of the chatbot.
        """

        raise NotImplementedError

    def sample(self,
               temperature = 0.8,
               top_k = 30,
               top_p = 0.95,
               repeat_penalty = 1.1,
               presence_penalty = 0.0):
        """
        Sample from current context and return a token.
        """

        raise NotImplementedError

    def send_message(self,
                     message,
                     stop_sequences = [],
                     n_tokens = 128,
                     ):
        """
        Send a message to the chatbot and return the response.
        """

        # If message is not empty
        if message != "":
            # Add message to context
            self.add_string_to_context(message)

            if __debug__:
                print(message, flush=True, end="")

        # Save state if necessary
        #@REVISIT I wonder if this should be moved out of this and implemented
        #@ in bot-aukerman or whatever library the user actually needs this for
        if not self.keep_response_in_context:
            self.save_state()

        # Generate response
        response_tokens = self.generate_tokens(n_tokens=n_tokens,
                                              stop_sequences=stop_sequences)

        # Restore state if necessary
        if not self.keep_response_in_context:
            self.restore_state()

        # Decode the generated response
        response_text = self.detokenize(response_tokens)

        # if __debug__:
        #     print("Response:", response_text)

        return response_text

    def generate_string(self, n_tokens = 128, stop_sequences = []):
        response_tokens = self.generate_tokens(n_tokens, stop_sequences)
        response_string = self.detokenize(response_tokens)

        return response_string

    def generate_tokens(self, n_tokens = 128, stop_sequences = []):
        # Generate one token at a time
        response_tokens = []
        response_text = "" #@REVISIT optimization? only use if regex is needed?

        # Parse stop_sequences into a dictionary of filter types
        #@REVISIT every time we parse?
        stop_filters = self.parse_stop_sequences(stop_sequences)

        # Generate tokens
        for i in range(n_tokens):
            next_token = self.sample(temperature=self.temperature,
                                     top_k=self.top_k,
                                     top_p=self.top_p,
                                     repeat_penalty=self.repeat_penalty,
                                     presence_penalty=self.presence_penalty)

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

    def increase_occurrence_count(self, token):
        if token in self.token_occurrence_count:
            self.token_occurrence_count[token] += 1
        else:
            self.token_occurrence_count[token] = 1

    def apply_penalties(self,
                        logits,
                        repeat_penalty = 1.0,
                        presence_penalty = 0.0):
        """
        Apply penalties to the logits.
        """

        #@TODO probably do something more like openai's range of -2 to 2

        if repeat_penalty > 1.0 or presence_penalty > 1.0:
            for token, count in self.token_occurrence_count.items():
                # Apply presence penalty
                if presence_penalty != 0.0:
                    logits[token] -= presence_penalty

                # Apply repetition penalty
                if repeat_penalty > 1.0: #@REVISIT allow less than one?
                    logits[token] /= repeat_penalty * count

        return logits


    def parse_stop_sequences(self, stop_sequences):
        """
        Parse a list of stop sequences into a dictionary of filter types.
        """

        stop_filters = {
            "strings": [],
            "token_sequences": [],
            "regexes": []
        }

        for stop_sequence in stop_sequences:
            # If stop sequence is a string
            if isinstance(stop_sequence, str):
                #@REVISIT this ugly architecture; remote chatbots don't need to
                #@ tokenize-- but the contents of the variables we're working
                #@ with feel unclear/unpredictable

                # If remote chatbot
                if self.is_remote:
                    # Just add string to token_sequences
                    stop_filters["token_sequences"].append(stop_sequence)

                # If local chatbot
                else:
                    # Tokenize the string
                    stop_tokens = self.tokenize(stop_sequence)

                    # Add tokens to token_sequences
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
        with open(f"{self.logdir}/{filename}",
                  mode = 'a+',
                  encoding = "utf-8") as logfile:
            logfile.write(message + "\n")

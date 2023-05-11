import sys
import subprocess
from llama_cpp import Llama
from .chatbot import Chatbot

#@REVISIT placement
# def progress_callback(progress):
#     print("Progress: {:.2f}%".format(progress * 100))
#     sys.stdout.flush()

class LlamaCPPChatbot(Chatbot):
    valid_options = ["model",
                     # "keep_context",
                     "keep_response_in_context"]
    model: Llama
    saved_states: list = []

    def __init__(self,
                 model_config: dict = {},
                 logdir = ""):

        super().__init__(model_config = model_config, logdir = "")

        self.keep_context = True #@TODO make option? any use cases?

        self.model = Llama(
            # Set the model path
            #@TODO-3 placeholder path
            model_path = '/mnt/Files/src/llama.cpp/models/gpt4all-7B/gpt4all-lora-unfiltered-converted.bin',
            # model_path = '/mnt/Files/src/llama.cpp/models/gpt4all-7B/gpt4all-lora-converted.bin',

            # Set context size
            n_ctx = 1024,

            # Set batch size
            n_batch = 512,

            # Set the seed
            seed = -1,

            # Set the number of threads
            n_threads = 8,

            #@TODO other params?
        )

        self.bos_token = self.model.token_bos()
        self.eos_token = self.model.token_eos()

    def tokenize(self, text: str):
        return self.model.tokenize(text.encode("utf-8"))

    def detokenize(self, tokens):
        return self.model.detokenize(tokens).decode("utf-8")

    def send_message(self,
                     message: str,
                     stop_sequences = [],
                     stop_regex = None,
                     n_tokens = 128):

        return super().send_message(message,
                                    stop_sequences = stop_sequences,
                                    stop_regex = stop_regex,
                                    n_tokens = n_tokens)

    def save_context(self):
        self.saved_states.append(self.model.save_state())

    def restore_context(self):
        saved_state = self.saved_states.pop()
        self.model.load_state(saved_state)

    def add_tokens_to_context(self, tokens):
        """
        Add tokens to the model's current context.
        """
        self.model.eval(tokens)

    def request_tokens(self, n_tokens = 128, stop_sequences = []):
        return super().request_tokens(n_tokens = n_tokens,
                                      stop_sequences = stop_sequences)

    def sample(self,
               temp = 0.8,
               top_k = 30,
               top_p = 0.95,
               repeat_penalty = 1.3):
        return self.model.sample(temp = temp,
                                 top_k = top_k,
                                 top_p = top_p,
                                 repeat_penalty = repeat_penalty)

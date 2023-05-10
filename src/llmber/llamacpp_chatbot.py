import sys
import subprocess
from llama_cpp import Llama
from .chatbot import Chatbot

#@REVISIT placement
# def progress_callback(progress):
#     print("Progress: {:.2f}%".format(progress * 100))
#     sys.stdout.flush()

class LlamaCPPChatbot(Chatbot):
    model: Llama
    saved_states: list = []

    def __init__(self,
                 model_config: dict = {},
                 logdir = ""):

        super().__init__(model_config = model_config, logdir = "")

        self.model = Llama(
            # Set the model path
            #@TODO-3 placeholder path
            model_path = '/mnt/Files/src/llama.cpp/models/gpt4all-7B/gpt4all-lora-unfiltered-converted.bin',
            # model_path = '/mnt/Files/src/llama.cpp/models/gpt4all-7B/gpt4all-lora-converted.bin',

            # Set context size
            n_ctx = 1024,

            # Set the seed
            seed = -1,

            # Set the number of threads
            n_threads = 8,
        )

        # # Reuse the last n tokens
        # repeat_last_n = 64

        # # Set the number of predictions
        # n_predict = 256


        # # Set batch size
        # n_batch = 8

        # # Set the top-k sampling
        # top_k = 40

        # # Set the top-p sampling
        # top_p = 0.9

        # # Set the temperature
        # temp = 0.8

        # # Set the repetition penalty
        # repeat_penalty = 1.3

    def send_message(self,
                     message: str,
                     stop_sequences = [],
                     stop_regex = None,
                     n_tokens = 128):

        # Tokenize message
        if message == "":
            message = " " #@SCAFFOLDING

        inputs = self.model.tokenize(message.encode("utf-8"))

        # Add tokens to context
        self.add_tokens_to_context(inputs)

        if __debug__:
            print(message, flush=True, end="")

        # Save context if necessary
        # if not self.keep_response_in_context:
        #     self.save_context()

        # Generate response
        response_tokens = self.request_tokens(n_tokens=n_tokens,
                                              stop_sequences=stop_sequences)

        # Restore context if necessary
        # if not self.keep_response_in_context:
        #     self.restore_context()

        # Decode the generated response
        response_text = self.model.detokenize(response_tokens).decode("utf-8")

        # if __debug__:
        #     print("Response:", response_text)

        return response_text

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
        # Generate one token at a time
        response_tokens = []
        response_text = "" #@REVISIT optimization? only use if regex is needed?

        # Parse stop_sequences into a dictionary of filter types
        # stop_filters = self.parse_stop_sequences(stop_sequences)

        for i in range(n_tokens):
            next_token = self.model.sample(temp=0.8,
                                           top_k=30,
                                           top_p=0.95,
                                           repeat_penalty=1.3)

            # next_token_id = next_token[0, -1].item()

            # Turn next_token into something that can be fed into the model
            # next_token = torch.tensor([[next_token_id]]).to(self.model.device)

            # Check if token is beginning-of-sequence token
            if next_token == self.model.token_bos():
                continue

            # Check if the token is end-of-sequence token
            if next_token == self.model.token_eos():
                break

            # Add next token to context
            self.add_tokens_to_context([next_token])

            # Add token to response
            response_tokens.append(next_token)

            # Decode the token and add to response text
            token_meaning = self.model.detokenize([next_token]).decode("utf-8")
            # print()
            # print(f"token_meaning: {token_meaning}")
            # print(f"next_token: {next_token}")
            # print(f"bos: {self.model.token_bos()}")
            # print(f"eos: {self.model.token_eos()}")
            response_text += token_meaning
            # break

            # Check for occurrences of stop sequences
            #@TODO-4
            # if self.check_stop_filters(stop_filters,
            #                            response_tokens,
            #                            response_text):
            #     break

            # Print the token
            if __debug__:
                print(token_meaning, flush=True, end="")

        if __debug__:
            print("", flush=True)
            sys.stdout.flush()

        return response_tokens

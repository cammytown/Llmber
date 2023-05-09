import sys
import subprocess
from llama_cpp import Llama
from .chatbot import Chatbot

#@REVISIT placement
def progress_callback(progress):
    print("Progress: {:.2f}%".format(progress * 100))
    sys.stdout.flush()

class LlamaCPPChatbot(Chatbot):
    model: Llama

    def __init__(self, name = "LlamaCPP"):
        super().__init__(name)

        raise Exception("LlamaCPPChatbot is not yet implemented.")

        self.model = Llama(
            # Set the model path
            # path_model = '/mnt/Files/src/llama.cpp/models/gpt4all-7B/gpt4all-lora-converted.bin'
            path_model = '/mnt/Files/src/llama.cpp/models/gpt4all-7B/gpt4all-lora-unfiltered-converted.bin'

            # Set the number of threads
            # n_threads = 8

            # # Reuse the last n tokens
            # repeat_last_n = 64

            # # Set the number of predictions
            # n_predict = 256

            # # Set context size
            # n_ctx = 1024

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

            # # Set the seed
            # seed = -1
        )

    #@REVISIT n_tokens and n_predict seem at odds; will be confusing
    def request_tokens(self, n_tokens = 256):
        token_string = ""

        # Sample (generate) tokens
        print("Sampling...")
        is_finished = False
        n_output = 0
        while n_output < n_tokens:
            #@REVISIT is this doing something important?
            self.model.eval()

            if self.model.has_unconsumed_input():
                self.model.ingest_all_pending_input()
            else:
                # Sample a token
                token = self.model.sample()

                # Convert the token to text
                text = self.model.token_to_str(token)

                # Add the token to the string
                token_string += text

                # Print the token
                print(text, end="", flush=True)

                n_output += 1

                # End of text token was found
                is_finished = token == self.model.token_eos()

            # If reverse prompt is encountered
            # if self.model.is_antiprompt_present()
            #     is_finished = True
        
            if is_finished:
                break

            # if self.model.is_finished()
            #     self.model.reset_remaining_tokens()
            #     is_interacting = True

        print("\n")

        # Flush stdout
        sys.stdout.flush()

        return token_string

    def send_message(self, message):
        # Generate tokens
        response_message = self.model()

        return response_message



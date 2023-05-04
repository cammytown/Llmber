import sys
import subprocess
import llamacpp
# from llama_cpp import Llama

from .Chatbot import Chatbot

#@REVISIT placement
def progress_callback(progress):
    print("Progress: {:.2f}%".format(progress * 100))
    sys.stdout.flush()

class LlamaCPPChatbot(Chatbot):
    model: llamacpp.LlamaInference

    def __init__(self, name = "LlamaCPP", model_config: dict = {}, logdir = ""):
        super().__init__(name, model_config = model_config, logdir = "")

        self.keep_context = True

        # Create the inference parameters
        params = llamacpp.InferenceParams.default_with_callback(progress_callback)

        # Set the model path
        # params.path_model = '/mnt/Files/src/llama.cpp/models/gpt4all-7B/gpt4all-lora-converted.bin'
        params.path_model = '/mnt/Files/src/llama.cpp/models/gpt4all-7B/gpt4all-lora-unfiltered-converted.bin'

        # Set the number of threads
        params.n_threads = 8

        # Reuse the last n tokens
        params.repeat_last_n = 64

        # Set the number of predictions
        params.n_predict = 256

        # Set context size
        params.n_ctx = 1024

        # Set batch size
        params.n_batch = 8

        # Set the top-k sampling
        params.top_k = 40

        # Set the top-p sampling
        params.top_p = 0.9

        # Set the temperature
        params.temp = 0.8

        # Set the repetition penalty
        params.repeat_penalty = 1.3

        # Set the seed
        params.seed = -1

        # Initialize the model
        self.model = llamacpp.LlamaInference(params)

        if not self.model: #@REVISIT does this do anything?
            raise Exception("Model failed to initialize")

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
        # Tokenize the message
        prompt_tokens = self.model.tokenize(message, True)

        # Supply the tokenized prompt
        self.model.update_input(prompt_tokens)

        # Ingest the prompt
        self.model.ingest_all_pending_input()

        # Generate tokens
        response_message = self.request_tokens()

        return response_message



import sys
import torch
import re
from typing import List
from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer

from .Chatbot import Chatbot

class HFTAutoBot(Chatbot):
    valid_options = ["name",
                     "keep_response_in_context"]

    logits: Optional[torch.Tensor] = None
    past_key_values: Optional[torch.Tensor] = None
    saved_contexts: List[tuple] = []

    def __init__(self,
                 name: str = "HFTAutoBot",
                 model_config: dict = {},
                 logdir: str = ""):

        super().__init__(name, model_config = model_config, logdir = logdir)

        self.keep_context = True

        if "name" not in model_config:
            raise ValueError("model_config must contain a 'name' key")

        model_name = model_config["name"].lower()

        cache_dir = "/run/media/cammy/PROJECTS2/huggingface_cache" #@SCAFFOLDING
        # cache_dir = None

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)

        # Set the model to half-precision floating point and move it to the GPU
        #@REVISIT make configurable?
        self.model.half().cuda()

        # Set the model to evaluation mode (disables dropout)
        self.model.eval()

    def send_message(self,
                     message,
                     stop_sequences = [],
                     stop_regex = None,
                     n_tokens = 128):

        # Tokenize message
        if message == "":
            message = " " #@SCAFFOLDING

        inputs = self.tokenizer.encode(message, return_tensors="pt") \
                .to(self.model.device)

        # Add tokens to context
        self.add_tokens_to_context(inputs)

        if __debug__:
            print(message, flush=True, end="")
            # print("inputs:", inputs)

        # Save context if necessary
        if not self.keep_response_in_context:
            self.save_context()

        # Generate response
        response = self.request_tokens(n_tokens=n_tokens,
                                       stop_sequences=stop_sequences)

        # Restore context if necessary
        if not self.keep_response_in_context:
            self.restore_context()

        # Decode the generated response
        response_text = self.tokenizer.decode(response, skip_special_tokens=True)

        # if __debug__:
        #     print("Response:", response_text)

        return response_text

    def save_context(self):
        self.saved_contexts.append((self.logits, self.past_key_values))

    def restore_context(self):
        self.logits, self.past_key_values = self.saved_contexts.pop()

    def add_tokens_to_context(self, tokens):
        """
        Add tokens to the model's current context.
        """

        with torch.no_grad():
            # Generate new logits
            outputs = self.model(tokens, past_key_values=self.past_key_values)

            # Update the logits and past key values
            self.logits = outputs.logits
            self.past_key_values = outputs.past_key_values

    def request_tokens(self, n_tokens = 128, stop_sequences = []):
        # Generate one token at a time
        response_tokens = []
        response_text = "" #@REVISIT optimization? only use if regex is needed?

        # Parse stop_sequences into a dictionary of filter types
        stop_filters = self.parse_stop_sequences(stop_sequences)

        with torch.no_grad():
            for i in range(n_tokens):
                next_token = self.sample(temperature=0.8, top_k=30, top_p=0.95)

                next_token_id = next_token[0, -1].item()

                # Turn next_token into something that can be fed into the model
                next_token = torch.tensor([[next_token_id]]).to(self.model.device)

                # Check if the token is an end-of-sequence token
                if next_token_id == self.tokenizer.eos_token_id:
                    break

                # Add next token to context
                self.add_tokens_to_context(next_token)

                # Add token to response
                response_tokens.append(next_token_id)

                # Decode the token
                token_meaning = self.tokenizer.decode(next_token_id,
                                                      skip_special_tokens=True)

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
                stop_tokens = self.tokenizer.encode(stop_sequence)
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

    def sample(self, temperature = 1.0, top_k = 0, top_p = 0.0):
        """
        Sample a token from the current logits.
        """

        # Apply temperature
        logits = self.logits / temperature

        #@TODO implement repetition/etc. penalty

        # Apply top-k
        if top_k > 0:
            top_k = min(top_k, logits.shape[-1])
            indices_to_remove = (logits < torch.topk(logits, top_k)[0][..., -1, None])
            logits[indices_to_remove] = -float("Inf")

        # Apply top-p
        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # Create mask to remove tokens
            mask = torch.zeros_like(logits, dtype=torch.bool)
            mask.scatter_(dim=-1,
                          index=sorted_indices,
                          src=sorted_indices_to_remove)

            # Set the logits to -inf
            logits[mask] = -float("Inf")

        # Sample from the distribution
        probs = torch.softmax(logits, dim=-1)

        # Squeeze the batch dimension
        probs = probs.squeeze(0)

        # Pick the next token
        token = torch.multinomial(probs, num_samples=1)

        # Add batch dimension
        token = token.unsqueeze(0)

        # Return the sampled token
        return token

    # #@REVISIT when they patch generate so it can return past_key_values
    # def request_tokens_generate(self, tokens, n_tokens = 128):
    #     # Generate one token at a time
    #     response = []

    #     with torch.no_grad():
    #         # Generate the response
    #         outputs = self.model.generate(input_ids=tokens,
    #                                       attention_mask=self.attention_mask,
    #                                       past_key_values=self.past_key_values,
    #                                       max_length=int(tokens.shape[1] + n_tokens),
    #                                       do_sample=True,
    #                                       top_k=50,
    #                                       top_p=0.95,
    #                                       temperature=0.7,
    #                                       num_return_sequences=1,
    #                                       pad_token_id=self.tokenizer.eos_token_id,
    #                                       # return_dict_in_generate=True,
    #                                       use_cache=True,
    #                                       )

    #         # self.add_tokens_to_context(outputs)

    #         # Update the logits and past key values
    #         self.logits = outputs.logits
    #         self.past_key_values = outputs.past_key_values

    #         # Convert the response to a list
    #         response = outputs[0].tolist()

    #     return response


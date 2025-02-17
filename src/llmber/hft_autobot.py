import sys
import torch
from typing import List, Tuple
from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer

from .chatbot import Chatbot

class HFTAutoBot(Chatbot):
    valid_options = ["model",
                     "temperature",
                     "top_k",
                     "top_p",
                     "repeat_penalty",
                     "presence_penalty",
                     "keep_response_in_context",
                     "use_cuda",
                     "max_context_length"]

    logits: Optional[torch.LongTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None
    saved_states: List[tuple] = []

    def __init__(self,
                 model_config: dict = {},
                 logdir: str = ""):

        super().__init__(name="HFTAutoBot",
                         model_config = model_config,
                         logdir = logdir)

        self.keep_context = True #@TODO make option?

        if "model" not in model_config:
            raise ValueError("model_config must contain a 'model' key")
        model_name = model_config["model"].lower()

        cache_dir = None

        print("Loading HFT tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                       cache_dir=cache_dir)

        print("Loading HFT model...")
        self.model = AutoModelForCausalLM.from_pretrained(model_name,
                                                          cache_dir=cache_dir)

        self.bos_token = self.tokenizer.bos_token_id
        self.eos_token = self.tokenizer.eos_token_id

        if "max_context_length" in model_config:
            if model_config["max_context_length"] > self.tokenizer.model_max_length:
                raise ValueError("max_context_length exceeds model's max length")

            self.max_context_length = model_config["max_context_length"]
        elif self.tokenizer.model_max_length is not None:
            #@TODO attempt to analyze hardware to determine max length
            self.max_context_length = self.tokenizer.model_max_length
        else:
            self.max_context_length = 512 #@REVISIT

        # If GPU is available
        if torch.cuda.is_available() \
                and self.model_config.get("use_cuda", True):
            if __debug__:
                print(f"DEBUG: CUDA is available. Moving model to GPU...", 
                      file=sys.stderr)
                if torch.cuda.device_count() > 0:
                    device_name = torch.cuda.get_device_name(0)
                    print(f"DEBUG: Using GPU: {device_name}", file=sys.stderr)
                print(f"DEBUG: Setting model to half precision...", file=sys.stderr)

            #@TODO what if GPU is insufficient?

            # Move the model to the GPU
            self.model.cuda()

            # Set the model to half-precision floating point
            self.model.half()
        else:
            if __debug__:
                print("DEBUG: Using CPU. CUDA availability: " + 
                      f"{torch.cuda.is_available()}, use_cuda setting: " +
                      f"{self.model_config.get('use_cuda', False)}", 
                      file=sys.stderr)

        if __debug__:
            device = next(self.model.parameters()).device
            print(f"DEBUG: Model is on device: {device}", file=sys.stderr)

        # Set the model to evaluation mode (disables dropout)
        self.model.eval()

        # Disable gradient calculation
        torch.set_grad_enabled(False)

    def tokenize(self, message):
        return self.tokenizer.encode(message, return_tensors="pt") \
                .to(self.model.device)

    def detokenize(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def get_state(self):
        return (self.logits, self.past_key_values)

    def set_state(self, state):
        self.logits, self.past_key_values = state

    def clear_state(self):
        self.logits = None
        self.past_key_values = None

    def add_tokens_to_context(self, tokens):
        """
        Add tokens to the model's current context.
        """

        # If tokens is list
        if isinstance(tokens, list): #@SCAFFOLDING
            # Turn tokens into something that can be fed into the model
            #@REVISIT I don't know why this is necessary
            #@REVISIT I don't really know when .to() is necessary/valuable
            tokens = torch.tensor([tokens]).to(self.model.device)

        # If context has max length
        if self.max_context_length is not None:
            # If input tokens exceed max context length
            if tokens.size(1) > self.max_context_length:
                # Truncate input tokens
                #@REVISIT have a truncate_tokens() method and combine with
                #@ truncate_past_key_values()?
                tokens = tokens[:, -self.max_context_length:]

            # If context is not empty
            elif self.past_key_values is not None:
                # Truncate past_key_values
                self.truncate_past_key_values(tokens.size(1))

        # Generate new logits
        outputs = self.model(tokens,
                             past_key_values=self.past_key_values)

        # Update the logits and past key values
        self.logits = outputs.logits
        self.past_key_values = outputs.past_key_values

    def truncate_past_key_values(self, buffer_size: int):
        """
        Truncate the past_key_values max length plus buffer size.
        """

        # Calculate sum of current context size and new tokens size
        sum_size = self.past_key_values[0][0].size(2) + buffer_size

        # If new context size will exceed max length
        if sum_size > self.max_context_length:
            # Calculate overage
            overage = sum_size - self.max_context_length

            # Reset past_key_values
            self.past_key_values = None

            # Re-add as much of the context as possible
            #@TODO hacky; attempting to truncate past_key_values manually
            #@ below led to incoherency; maybe I did it wrong
            self.add_string_to_context(self.context_string[-overage:])

            # Remove overage from past_key_values
            # new_past_key_values = []

            # for layer in self.past_key_values:
            #     new_layer = []
            #     for tensor in layer:
            #         tensor = tensor[:, :, overage:, :]
            #         new_layer.append(tensor)

            #     new_past_key_values.append(new_layer)

            # self.past_key_values = new_past_key_values

    def send_message(self,
                     message,
                     stop_sequences = [],
                     n_tokens = 128):
        return super().send_message(message,
                             stop_sequences = stop_sequences,
                             n_tokens = n_tokens)

    def generate_tokens(self, n_tokens = 128, stop_sequences = []):
        return super().generate_tokens(n_tokens = n_tokens,
                                      stop_sequences = stop_sequences)

    def sample(self,
               temperature = 1.0,
               top_k = 0,
               top_p = 0.0,
               repeat_penalty = 1.1,
               presence_penalty = 0.0) -> int:
        """
        Sample a token from the current logits.
        """

        if self.logits is None:
            # If no context, add random character to context
            #@REVISIT what should we do here? probably something smarter
            self.add_string_to_context("A");

        # Apply temperature
        logits = self.logits / temperature

        #@TODO
        # for banned_token in self.args.token_ban:
        #     self.logits[banned_token] = -float('inf')

        # Apply repetition penalty
        #@REVISIT passing in logits[0, -1]; is it working?
        self.apply_penalties(logits[0, -1], repeat_penalty, presence_penalty)

        # Apply top-k
        if top_k > 0:
            self.apply_top_k(logits, top_k)

        # Apply top-p
        if top_p > 0.0:
            self.apply_top_p(logits, top_p)

        # Convert logits to probabilities
        probs = torch.softmax(logits, dim=-1)

        # Squeeze (remove) the batch dimension
        probs = probs.squeeze(0)

        # Pick the next token
        token_tensor = torch.multinomial(probs, num_samples=1).long()
        token = token_tensor[-1].item()

        # Increase token occurrence count
        self.increase_occurrence_count(token)

        # Return the sampled token
        return token

    def apply_top_k(self, logits, top_k = 0.0):
        if top_k > 0:
            top_k = min(top_k, logits.shape[-1])
            indices_to_remove = (logits < torch.topk(logits, top_k)[0][..., -1, None])
            logits[indices_to_remove] = -float("Inf")

    def apply_top_p(self, logits, top_p = 0.0):
        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits,
                                                       descending=True,
                                                       dim=-1)

            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1),
                                            dim=-1)

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

    # #@REVISIT when they patch generate so it can return past_key_values
    # def generate_tokens_generate(self, tokens, n_tokens = 128):
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


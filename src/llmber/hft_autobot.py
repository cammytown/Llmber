import sys
import torch
from typing import List
from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer

from .chatbot import Chatbot

class HFTAutoBot(Chatbot):
    valid_options = ["model",
                     "keep_response_in_context"]

    logits: Optional[torch.Tensor] = None
    past_key_values: Optional[torch.Tensor] = None
    saved_contexts: List[tuple] = []

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

        cache_dir = "/run/media/cammy/PROJECTS2/huggingface_cache" #@SCAFFOLDING
        # cache_dir = None

        self.tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                       cache_dir=cache_dir)

        self.model = AutoModelForCausalLM.from_pretrained(model_name,
                                                          cache_dir=cache_dir)

        self.bos_token = self.tokenizer.bos_token_id
        self.eos_token = self.tokenizer.eos_token_id

        # Set the model to half-precision floating point and move it to the GPU
        #@REVISIT make configurable?
        self.model.half().cuda()

        # Set the model to evaluation mode (disables dropout)
        self.model.eval()

        # Disable gradient calculation
        torch.set_grad_enabled(False)

    def tokenize(self, message):
        return self.tokenizer.encode(message, return_tensors="pt") \
                .to(self.model.device)

    def detokenize(self, tokens):
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def get_context(self):
        return (self.logits, self.past_key_values)

    def set_context(self, context):
        self.logits, self.past_key_values = context

    def add_tokens_to_context(self, tokens):
        """
        Add tokens to the model's current context.
        """

        # If tokens is list
        if isinstance(tokens, list): #@SCAFFOLDING
            # Turn next_token into something that can be fed into the model
            #@REVISIT I don't know why this is necessary
            #@REVISIT I don't really know when .to() is necessary
            tokens = torch.tensor([tokens]).to(self.model.device)

        # Generate new logits
        outputs = self.model(tokens,
                             past_key_values=self.past_key_values)

        # Update the logits and past key values
        self.logits = outputs.logits
        self.past_key_values = outputs.past_key_values

    def send_message(self,
                     message,
                     stop_sequences = [],
                     stop_regex = None,
                     n_tokens = 128):
        return super().send_message(message,
                             stop_sequences = stop_sequences,
                             stop_regex = stop_regex,
                             n_tokens = n_tokens)

    def request_tokens(self, n_tokens = 128, stop_sequences = []):
        return super().request_tokens(n_tokens = n_tokens,
                                      stop_sequences = stop_sequences)

    def sample(self,
               temp = 1.0,
               top_k = 0,
               top_p = 0.0,
               repeat_penalty = 0) -> int:
        """
        Sample a token from the current logits.
        """

        # Apply temperature
        logits = self.logits / temp

        #@TODO-4 implement repetition/etc. penalty

        # Apply top-k
        if top_k > 0:
            top_k = min(top_k, logits.shape[-1])
            indices_to_remove = (logits < torch.topk(logits, top_k)[0][..., -1, None])
            logits[indices_to_remove] = -float("Inf")

        # Apply top-p
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

        # Convert logits to probabilities
        probs = torch.softmax(logits, dim=-1)

        # Squeeze (remove) the batch dimension
        probs = probs.squeeze(0)

        # Pick the next token
        token_tensor = torch.multinomial(probs, num_samples=1).long()
        token = token_tensor[-1].item()

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


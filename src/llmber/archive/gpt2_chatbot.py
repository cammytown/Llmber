import sys
import torch
from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer

from .chatbot import Chatbot

class HuggingFaceAutoChatbot(Chatbot):
    logits: Optional[torch.Tensor] = None
    past_key_values: Optional[torch.Tensor] = None

    def __init__(self, model = "GPT2"):
        super().__init__(name)

        self.keep_context = True

        # model_name = "PygmalionAI/pygmalion-350m"
        # model_name = "gpt2-large"
        model_name = model.lower()
        cache_dir = "/run/media/cammy/PROJECTS2/huggingface_cache" #@SCAFFOLDING

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)

        self.model.half().cuda()

        # Set the model to evaluation mode (disables dropout)
        self.model.eval()

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

    def send_message(self, message, stop_sequences = []):
        # Tokenize message
        if message == "":
            message = " " #@SCAFFOLDING

        inputs = self.tokenizer.encode(message, return_tensors="pt").to(self.model.device)

        # Add tokens to context
        self.add_tokens_to_context(inputs)
        # self.add_tokens_to_context(inputs[0])

        if __debug__:
            print(message, flush=True, end="")
            # print("inputs:", inputs)

        # Generate response
        response = self.request_tokens(n_tokens=128,
                                       stop_sequences=stop_sequences)

        # Decode the generated response
        response_text = self.tokenizer.decode(response, skip_special_tokens=True)

        # if __debug__:
        #     print("Response:", response_text)

        return response_text

    def request_tokens(self, n_tokens = 128, stop_sequences = []):
        # Generate one token at a time
        response = []

        # Convert stop sequences to token IDs
        stop_token_sequences = []
        for stop_sequence in stop_sequences:
            stop_token_sequences.append(self.tokenizer.encode(stop_sequence))

        with torch.no_grad():
            for i in range(n_tokens):
                next_token = self.sample(temperature=0.8, top_k=30, top_p=0.95)

                next_token_id = next_token[0, -1].item()

                # Turn next_token into something that can be fed into the model
                next_token = torch.tensor([[next_token_id]]).to(self.model.device)

                # Add next token to context
                self.add_tokens_to_context(next_token)

                # Add token to response
                response.append(next_token_id)

                # Check if the token is an end-of-sequence token
                if next_token_id == self.tokenizer.eos_token_id:
                    break

                # Check for occurrences of stop sequences
                stop_sequence_found = False
                for stop_token_sequence in stop_token_sequences:
                    if len(response) >= len(stop_token_sequence):
                        if response[-len(stop_token_sequence):] == stop_token_sequence:
                            stop_sequence_found = True
                            break
                if stop_sequence_found:
                    break

                # Print the token
                if __debug__:
                    token_meaning = self.tokenizer.decode(next_token_id,
                                                          skip_special_tokens=True)
                    print(token_meaning, flush=True, end="")

        if __debug__:
            print("", flush=True)
            sys.stdout.flush()

        return response

    def sample(self, temperature = 1.0, top_k = 0, top_p = 0.0):
        """
        Sample a token from the current logits.
        """

        # Apply temperature
        logits = self.logits / temperature

        #@TODO implement reptition/etc. penalty

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


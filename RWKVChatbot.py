import os
import sys
import re

# set these before import RWKV
os.environ['RWKV_JIT_ON'] = '1'
os.environ["RWKV_CUDA_ON"] = '1' # '1' to compile CUDA kernel (10x faster), requires c++ compiler & cuda libraries

# Strategy Examples: (device = cpu/cuda/cuda:0/cuda:1/...)
# 'cpu fp32' = all layers cpu fp32
# 'cuda fp16' = all layers cuda fp16
# 'cuda fp16i8' = all layers cuda fp16 with int8 quantization
# 'cuda fp16i8 *10 -> cpu fp32' = first 10 layers cuda fp16i8, then cpu fp32 (increase 10 for better speed)
# 'cuda:0 fp16 *10 -> cuda:1 fp16 *8 -> cpu fp32' = first 10 layers cuda:0 fp16, then 8 layers cuda:1 fp16, then cpu fp32
#
# Basic Strategy Guide: (fp16i8 works for any GPU)
# 100% VRAM = 'cuda fp16'                   # all layers cuda fp16
#  98% VRAM = 'cuda fp16i8 *1 -> cuda fp16' # first 1 layer  cuda fp16i8, then cuda fp16
#  96% VRAM = 'cuda fp16i8 *2 -> cuda fp16' # first 2 layers cuda fp16i8, then cuda fp16
#  94% VRAM = 'cuda fp16i8 *3 -> cuda fp16' # first 3 layers cuda fp16i8, then cuda fp16
#  ...
#  50% VRAM = 'cuda fp16i8'                 # all layers cuda fp16i8
#  48% VRAM = 'cuda fp16i8 -> cpu fp32 *1'  # most layers cuda fp16i8, last 1 layer  cpu fp32
#  46% VRAM = 'cuda fp16i8 -> cpu fp32 *2'  # most layers cuda fp16i8, last 2 layers cpu fp32
#  44% VRAM = 'cuda fp16i8 -> cpu fp32 *3'  # most layers cuda fp16i8, last 3 layers cpu fp32
#  ...
#   0% VRAM = 'cpu fp32'                    # all layers cpu fp32
#r
# Use '+' for STREAM mode, which can save VRAM too, and it is sometimes faster
# 'cuda fp16i8 *10+' = first 10 layers cuda fp16i8, then fp16i8 stream the rest
# to it (increase 10 for better speed)
#
# Extreme STREAM: 3G VRAM is enough to run RWKV 14B (slow. will be faster in future)
# 'cuda fp16i8 *0+ -> cpu fp32 *1' = stream all layers cuda fp16i8, last 1 layer [ln_out+head] cpu fp32
#

from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS

from .Chatbot import Chatbot

class RWKVChatbot(Chatbot):
    model: RWKV
    pipeline: PIPELINE
    args: PIPELINE_ARGS #@REVISIT naming

    token_occurrence_count: dict = {} #@REVISIT should this be class level or method?
    logits: list
    state = None

    # all_tokens: list = [] #@REVISIT naming

    def __init__(self, name: str = "RWKV", model_config: dict = {}, logdir: str = ""):
        super().__init__(name, model_config=model_config, logdir=logdir)

        self.keep_context = True

        model_path = '/run/media/cammy/PROJECTS2/models/rwkv-4-raven/RWKV-4-Raven-1B5-v9-Eng99%-Other1%-20230411-ctx4096.pth'
        # model_path = '/run/media/cammy/PROJECTS2/models/rwkv-4-raven/RWKV-4-Raven-7B-v9-Eng99%-Other1%-20230412-ctx8192.pth'
        # model_path = '/run/media/cammy/PROJECTS2/models/rwkv-4-pile/RWKV-4-Pile-7B-20230406-ctx8192-test949.pth'

        #'/run/media/cammy/PROJECTS2/models/rwkv-4-raven/RWKV-4-Raven-14B-v9-Eng99%-Other1%-20230412-ctx8192.pth'

        tokenizer_path = "/home/cammy/Fsrc/ChatRWKV/20B_tokenizer.json"

        # Initialize the model
        self.model = RWKV(model=model_path, strategy='cuda fp16')
        # self.model = RWKV(model=model_path, strategy='cuda fp16i8 -> cpu fp32 *3')
        self.pipeline = PIPELINE(self.model, tokenizer_path)

        self.args = PIPELINE_ARGS(
                # variability of the generated tokens
                temperature = 1.5,

                # select tokens with prob that can add to
                # another token prob to equal >= top_p
                top_p = 0.2,

                # select from top_k most likely tokens
                top_k = 100,

                # frequency penalty
                alpha_frequency = 0.25,

                # presence penalty
                alpha_presence = 0.25,

                # ban the generation of some tokens
                token_ban = [0],

                # stop generation on these tokens
                token_stop = [],

                # split input into chunks to save VRAM
                # (shorter -> slower)
                chunk_len = 256)

        if not self.model: #@REVISIT does this do anything?
            raise Exception("Model failed to initialize")

    def add_tokens_to_state(self, tokens):
        # Decode tokens and add to context
        #@REVISIT optimization possibilities
        if __debug__:
            token_meaning = self.pipeline.decode(tokens)
            # self.context += token_meaning
            print(token_meaning, end='', flush=True) #@SCAFFOLDING

        # Add tokens to state
        while len(tokens) > 0:
            self.logits, self.state = self.pipeline.model.forward(tokens[:self.args.chunk_len], self.state)
            tokens = tokens[self.args.chunk_len:]

    def add_string_to_state(self, string):
        tokens = self.pipeline.encode(string)
        self.add_tokens_to_state(tokens)

    def send_message(self, message, stop_sequences = [], stop_regex = None):
        # Encode message
        tokens = self.pipeline.encode(message)

        # Add tokens to state
        if __debug__:
            print("Updating model state with tokens: {}".format(tokens))
        self.add_tokens_to_state(tokens)

        # Generate response
        response_message = self.request_tokens(n_tokens=256,
                                               stop_sequences=stop_sequences,
                                               stop_regex=stop_regex)

        return response_message

    #@REVISIT n_tokens and n_predict seem at odds; will be confusing
    #@REVISIT currently returns string, not list of tokens; confusing?
    def request_tokens(self,
                       n_tokens = 256,
                       stop_sequences = [],
                       stop_regex = None):

        token_string = ""

        for i in range(n_tokens):
            for n in self.args.token_ban:
                self.logits[n] = -float('inf')
            for n in self.token_occurrence_count:
                self.logits[n] -= (self.args.alpha_presence + self.token_occurrence_count[n] * self.args.alpha_frequency)

            # sampler
            token = self.pipeline.sample_logits(self.logits,
                                       temperature=self.args.temperature,
                                       top_p=self.args.top_p,
                                       top_k=self.args.top_k)

            # If token is in token_stop, stop generation
            if token in self.args.token_stop:
                break

            # Increment self.token_occurrence_count of token
            if token not in self.token_occurrence_count:
                self.token_occurrence_count[token] = 1
            else:
                self.token_occurrence_count[token] += 1

            self.add_tokens_to_state([token])

            # Decode token and add to token_string
            token_string += self.pipeline.decode([token])

            # Check for occurrence of a stop sequence:
            if self.check_for_stop(token_string, stop_sequences, stop_regex):
                break

            # tmp = self.pipeline.decode(self.all_tokens[out_last:])
            # if '\ufffd' not in tmp: # is valid utf-8 string?
            #     # if callback:
            #     #     callback(tmp)
            #     print(tmp, end='', flush=True)
            #     out_str += tmp
            #     out_last = len(self.all_tokens)
            #     # out_last = i + 1

        if __debug__:
            # Flush stdout
            sys.stdout.flush()
            # print(f"Generated {len(token_string)} tokens")

        return token_string

    def check_for_stop(self, token_string, stop_sequences, stop_regex):
        # Check for occurrence of a stop sequence:
        for stop_sequence in stop_sequences:
            #@REVISIT I check the whole string because I worry that the
            #stop sequence might occur in the middle of a token
            if stop_sequence in token_string:
                if __debug__:
                    print(f"Found stop sequence {stop_sequence} in {token_string}")
                return True

        # Check for occurrence of a stop regex:
        #@REVISIT doing this for every token is questionable
        if stop_regex:
            if re.search(stop_regex, token_string):
                if __debug__:
                    print(f"Found stop regex {stop_regex} in {token_string}")
                return True

import os
import sys

# set these before import RWKV
os.environ['RWKV_JIT_ON'] = '1'
os.environ["RWKV_CUDA_ON"] = '1' # '1' to compile CUDA kernel (10x faster), requires c++ compiler & cuda libraries

########################################################################################################
#
# Use '/' in model path, instead of '\'. Use ctx4096 models if you need long ctx.
#
# fp16 = good for GPU (!!! DOES NOT support CPU !!!)
# fp32 = good for CPU
# bf16 = worse accuracy, supports CPU
# xxxi8 (example: fp16i8, fp32i8) = xxx with int8 quantization to save 50% VRAM/RAM, slower, slightly less accuracy
#
# We consider [ln_out+head] to be an extra layer, so L12-D768 (169M) has "13" layers, L24-D2048 (1.5B) has "25" layers, etc.
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
# 'cuda fp16i8 *10+' = first 10 layers cuda fp16i8, then fp16i8 stream the rest to it (increase 10 for better speed)
#
# Extreme STREAM: 3G VRAM is enough to run RWKV 14B (slow. will be faster in future)
# 'cuda fp16i8 *0+ -> cpu fp32 *1' = stream all layers cuda fp16i8, last 1 layer [ln_out+head] cpu fp32
#
# ########################################################################################################

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

    all_tokens: list = [] #@REVISIT naming

    def __init__(self, name = "RWKVChatbot"):
        super().__init__(name)

        # Initialize the model
        # model = RWKV(model='/run/media/cammy/PROJECTS2/models/rwkv-4-raven/RWKV-4-Raven-14B-v9-Eng99%-Other1%-20230412-ctx8192.pth', strategy='cpu fp32')
        self.model = RWKV(model='/run/media/cammy/PROJECTS2/models/rwkv-4-raven/RWKV-4-Raven-1B5-v9-Eng99%-Other1%-20230411-ctx4096.pth', strategy='cuda fp16')
        self.pipeline = PIPELINE(self.model, "/home/cammy/Fsrc/ChatRWKV/20B_tokenizer.json") # 20B_tokenizer.json is in https://github.com/BlinkDL/ChatRWKV

        self.args = PIPELINE_ARGS(temperature = 1.0, top_p = 0.7, top_k = 100, # top_k = 0 then ignore
                             alpha_frequency = 0.25,
                             alpha_presence = 0.25,
                             token_ban = [0], # ban the generation of some tokens
                             token_stop = [], # stop generation whenever you see any token here
                             chunk_len = 256) # split input into chunks to save VRAM (shorter -> slower)

        if not self.model: #@REVISIT does this do anything?
            raise Exception("Model failed to initialize")

    def add_tokens_to_state(self, tokens):
        # Add tokens to all_tokens
        self.all_tokens += tokens

        # Add tokens to state
        while len(tokens) > 0:
            self.logits, self.state = self.pipeline.model.forward(tokens[:self.args.chunk_len], self.state)
            tokens = tokens[self.args.chunk_len:]

    def send_message(self, message):
        # Encode message
        tokens = self.pipeline.encode(message)

        # Add tokens to state
        self.add_tokens_to_state(tokens)

        self.logits, self.state = self.pipeline.model.forward(tokens, self.state)

        # Generate response
        response_message = self.request_tokens()

        return response_message

    #@REVISIT n_tokens and n_predict seem at odds; will be confusing
    #@REVISIT currently returns string, not list of tokens; confusing?
    def request_tokens(self, n_tokens = 256):
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

            # Decode token
            token_meaning = self.pipeline.decode([token])
            token_string += token_meaning
            print(token_meaning, end='', flush=True)

            # tmp = self.pipeline.decode(self.all_tokens[out_last:])
            # if '\ufffd' not in tmp: # is valid utf-8 string?
            #     # if callback:
            #     #     callback(tmp)
            #     print(tmp, end='', flush=True)
            #     out_str += tmp
            #     out_last = len(self.all_tokens)
            #     # out_last = i + 1

        # Flush stdout
        sys.stdout.flush()

        return token_string

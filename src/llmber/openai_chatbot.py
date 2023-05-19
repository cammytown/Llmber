import os
from enum import Enum
from collections import Counter
import openai
import json

from .chatbot import Chatbot

class OpenAIChatbot(Chatbot):
    valid_options = ["model",
                     "keep_context",
                     "keep_response_in_context"]

    api = openai

    # Completion uses context, Chat uses message_history #@REVISIT architecture
    context = ""
    message_history = []

    ModelType = Enum('Mode', ['completion', 'chat'])
    model_type: ModelType

    def __init__(self,
                 name = "OpenAI",
                 model_config = { "model": "text-ada-001" },
                 logdir = ""):

        super().__init__(name = name,
                         model_config = model_config,
                         logdir = logdir)

        self.is_remote = True
        self.keep_context = False

        openai_key = self.retrieve_key('openai')
        if openai_key:
            self.api.api_key = openai_key
        else:
            raise Exception("Couldn't retrieve OpenAI key")

        self.model_config = model_config

        #@TODO we should probably first check a list of valid models and then
        #@ warn user if they're using an unknown model before falling back on this
        if model_config["model"].startswith('text-'):
            self.model_type = self.ModelType.completion
        elif model_config["model"].startswith('gpt-'):
            self.model_type = self.ModelType.chat

        # self.openai_bot = OpenAIEngine(openai_key)
        # self.openai_bot.set_engine('davinci')
        # self.openai_bot.set_temperature(0.7)
        # self.openai_bot.set_max_tokens(100)
        # self.openai_bot.set_top_p(1)
        # self.openai_bot.set_frequency_penalty(0)
        # self.openai_bot.set_presence_penalty(0)
        # self.openai_bot.set_stop(['\n', ' Human:', ' AI:'])

    def send_message(self,
                     message,
                     stop_sequences = [],
                     n_tokens = 128):

        # if __debug__:
        #     print("Sending message to OpenAI: {}".format(message))

        if self.model_type == self.ModelType.completion:
            return self.send_completion(message, n_tokens, stop_sequences)
        elif self.model_type == self.ModelType.chat:
            return self.send_chat_message(message, n_tokens, stop_sequences)
        else:
            raise Exception("Unknown OpenAI model type")

    def send_completion(self, message, n_tokens = 128, stop_sequences = []):
        if self.keep_context:
            # Add message to context
            self.context += message
        else:
            # Replace context with message
            self.context = message

        # print("=== OpenAI context ===")
        # print(self.context)
        # print("--- END OpenAI message ---")

        # Filter out invalid stop sequences
        stop_sequences = self.filter_valid_openai_stops(stop_sequences)

        # Send message to OpenAI
        response_obj = openai.Completion.create(
            model = self.model_config['model'],
            prompt = self.context,

            # Maximum number of tokens to generate
            max_tokens = n_tokens,

            # Variety of possible tokens
            # (OpenAI says temperature or top_p, not both)
            temperature = 0.7,
            # top_p = 1,

            # Penalize new tokens based on whether they appear in the text so far
            # frequency_penalty = 
            # presence_penalty = 

            # Number of completions (i.e. number of choices) to return
            # (This will naturally increase token usage)
            # n = 1,

            # Stop sequence(s)
            # stop = ['\n']
        )

        print("=== OpenAI response ===")
        print(response_obj)

        # Parse response
        response_message = response_obj['choices'][0]['text']

        # Add response to context if configured to
        if self.keep_response_in_context:
            self.context += response_message

        # Log usage
        self.log_usage(response_obj)

        return response_message

    def send_chat_message(self, message, n_tokens = 128, stop_sequences = []):
        raise Exception("Chat model not implemented yet")

        self.message_history.append({"role": "user", "content": message})

        response_obj = openai.ChatCompletion.create(
            model = self.model_config['model'],
            messages = self.message_history,
            max_tokens = n_tokens,
        )

        response_message = response_obj['choices'][0]['message']

        self.message_history.append({"role": "assistant", "content": response_message})

        # Log usage
        self.log_usage(response_obj)

        # Return response
        return response_message

    # def send_chat_message(self, user_message, system_message = None):
    #     if not system_message:
    #         system_message = "You are a helpful assistant."

    #     response_obj = openai.ChatCompletion.create(
    #       model=self.model_config['model'],
    #       messages=[
    #             {"role": "system", "content": system_message},
    #             {"role": "user", "content": user_message},
    #         ]
    #     )

    #     response_message = response_obj['choices'][0]['message']['content']

    #     # Log usage
    #     self.log_usage(response_obj)

    #     # Return response
    #     return response_message

    #@REVISIT architecture; maybe place in base class if it's generic enough
    def filter_valid_openai_stops(self, stop_sequences: list):
        valid_stops = []
        max_openai_stops = 4

        # Check stop sequences
        for stop_sequence in stop_sequences:
            # If stop_sequence is string
            if isinstance(stop_sequence, str):
                # Add to valid_stops
                valid_stops.append(stop_sequence)

                # If valid_stops reaches max_openai_stops, break
                if len(valid_stops) >= max_openai_stops:
                    break

            else:
                # Ignore non-string stop sequence #@REVISIT
                print(f"WARNING: ignoring non-string stop: {stop_sequence}")

        return valid_stops

    def log_usage(self, response_obj):
        filename = f"{self.logdir}/ChatGPT/request-usage.txt"
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, 'w+', encoding="utf-8") as file:
            if file.read() == '':
                #@ is dict() necessary?
                total_usage = Counter(response_obj['usage'])
            else:
                # Read file
                previous_usage = Counter(json.load(file))

                # Combine previous usage with query usage
                request_usage = Counter(response_obj['usage'])
                total_usage = previous_usage + request_usage

            # Reset file read cursor to start
            file.seek(0)

            # Write new total to file
            json.dump(dict(total_usage), file)

            # Remove file content after the cursor we just wrote up to
            file.truncate()

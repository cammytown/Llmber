import os
from enum import Enum
from collections import Counter
import openai
import json

from .Chatbot import Chatbot

class OpenAIChatbot(Chatbot):
    model_config: dict
    api = openai

    # Completion uses context, Chat uses message_history #@REVISIT architecture
    context = ""
    message_history = []

    ModelType = Enum('Mode', ['completion', 'chat'])
    model_type: ModelType

    def __init__(self, name = "OpenAI", model_config = { "name": "text-ada-001" }):
        super().__init__(name)

        openai_key = self.retrieve_key('openai')
        if openai_key:
            self.api.api_key = openai_key
        else:
            raise Exception("Couldn't retrieve OpenAI key")

        self.model_config = model_config

        #@TODO we should probably first check a list of valid models and then
        #@ warn user if they're using an unknown model before falling back on this
        if model_config.name.startswith('text-'):
            self.model_type = self.ModelType.completion
        elif model_config.name.startswith('gpt-'):
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
                     stop_regex = None,
                     n_tokens = 128):

        if __debug__:
            print("Sending message to OpenAI: {}".format(message))

        if self.model_type == self.ModelType.completion:
            return self.send_completion(message, n_tokens)
        elif self.model_type == self.ModelType.chat:
            return self.send_chat_message(message)
        else:
            raise Exception("Unknown OpenAI model type")

    def send_completion(self, message, n_tokens = 128):
        # Add message to context
        self.context += message

        # Send message to OpenAI
        response_obj = openai.Completion.create(
            model = self.model_config['name'],
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

        print("OpenAI response:")
        print(response_obj)

        # Parse response
        response_message = response_obj['choices'][0]['text']

        # Add response to context if configured to
        if self.keep_response_in_context:
            self.context += response_message

        # Log usage
        self.log_usage(response_obj)

        return response_message

    def send_chat_message(self, message):
        self.message_history.append({"role": "user", "content": message})

        response_obj = openai.ChatCompletion.create(
            model = self.model_config['name'],
            messages = self.message_history
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
    #       model=self.model_config['name'],
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

    def log_usage(self, response_obj):
        filename = f"{self.logdir}/ChatGPT/request-usage.txt"
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, 'w+') as file:
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

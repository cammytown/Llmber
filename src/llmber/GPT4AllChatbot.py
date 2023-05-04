from .Chatbot import Chatbot
from nomic.gpt4all import GPT4All

class GPT4AllChatbot(Chatbot):
    api: GPT4All

    def __init__(self, name = "GPT4All"):
        super().__init__(name)
        api = GPT4All()
        api.open()

    def send_message(self, message):
        response = self.api.ask(message)
        api.prompt('write me a story about a lonely computer')

        response_message = response['content']

        return response_message


import Bard
from .Chatbot import Chatbot

class BardChatbot(Chatbot):
    api: Bard.Chatbot

    def __init__(self, name = "Bard", model_config = {}, logdir = ""):
        super().__init__(name)

        self.keep_context = True

        # Retrieve Google Bard key:
        googlebard_key = self.retrieve_key('googlebard')
        if googlebard_key:
            #@REVISIT-4 connect once at start or once per request?
            self.api = Bard.Chatbot(googlebard_key)
        else:
            print("ERROR: Couldn't retrieve GoogleBard key")

    def send_message(self, message):
        response = self.api.ask(message)

        # Extract bot's message to user:
        #@REVISIT also consider incorporating response['choices']
        #which includes an array of other possible responses.

        response_message = response['content']

        # with open("chatbot-data/GoogleBard/interaction-log.txt", "a") as file:
        #     file.write(json.dumps(interaction) + "\n")

        return response_message
        return response

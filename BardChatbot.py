import Bard

class BardChatbot(Chatbot):
    api: Bard.Chatbot

    def __init__(self, name = "Bard"):
        super().__init__(name)

        # Retrieve Google Bard key:
        googlebard_key = self.retrieve_key('googlebard')
        if googlebard_key:
            #@REVISIT-4 connect once at start or once per request?
            self.api = Bard.Chatbot(googlebard_key)
        else:
            print("ERROR: Couldn't retrieve GoogleBard key")

    def send_message(self, message):
        response = self.api.ask(user_message)

        # Extract bot's message to user:
        #@REVISIT also consider incorporating response['choices']
        #which includes an array of other possible responses.

        response_message = response['content']

        # with open("chatbots/GoogleBard/interaction-log.txt", "a") as file:
        #     file.write(json.dumps(interaction) + "\n")

        return response_message
        return response

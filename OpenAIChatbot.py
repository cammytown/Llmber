import openai

class OpenAIChatbot(Chatbot):
    api = openai

    def __init__(self, name = "ChatGPT"):
        super().__init__(name)

        openai_key = self.retrieve_key('openai')
        if openai_key:
            self.api.api_key = openai_key
        else:
            print("ERROR: Couldn't retrieve OpenAI key")

        # self.openai_bot = OpenAIEngine(openai_key)
        # self.openai_bot.set_engine('davinci')
        # self.openai_bot.set_temperature(0.7)
        # self.openai_bot.set_max_tokens(100)
        # self.openai_bot.set_top_p(1)
        # self.openai_bot.set_frequency_penalty(0)
        # self.openai_bot.set_presence_penalty(0)
        # self.openai_bot.set_stop(['\n', ' Human:', ' AI:'])

    def send_message(self, user_message):
        # Send dialogue history to chatbot:
        response = self.api.send_request(user_message)

        # Return response:
        return response

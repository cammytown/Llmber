from .chatbot import Chatbot

class RemoteChatbot(Chatbot):
    context = ""

    def __init__(self,
                 name: str = "RemoteChatbot",
                 model_config: dict = {},
                 logdir: str = ""):

        super().__init__(name = name,
                         model_config = model_config,
                         logdir = logdir)

        self.is_remote = True

    def add_string_to_context(self, string):
        self.context += string

    def get_context(self):
        return self.context

    def set_context(self, context):
        self.context = context

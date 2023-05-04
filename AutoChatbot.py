from .Chatbot import Chatbot

class AutoChatbot(Chatbot):
    """
    A chatbot that can be used to send messages to and receive messages from
    a LLM. Automatically selects chatbot based on model_config.
    """

    chatbot: Chatbot

    def __init__(self,
                 name = "AutoChatbot",
                 model_config: dict = { "name": "gpt2" }):

        super().__init__(name, model_config = model_config)

        # If model is to be run locally
        if "remote" not in model_config or not model_config["remote"]:
            match model_config["name"].lower():
                case "gpt2" | "pygmalion" | "decapoda-research/llama-7b-hf":
                    from .HFTAutoBot import HFTAutoBot
                    self.chatbot = HFTAutoBot(model_config = model_config)
                case "rwkv":
                    from .RWKVChatbot import RWKVChatbot
                    self.chatbot = RWKVChatbot(model_config=model_config)
                case "llamacpp":
                    from .LlamaCPPChatbot import LlamaCPPChatbot
                    self.chatbot = LlamaCPPChatbot(model_config=model_config)
                case _:
                    raise ValueError(f"Invalid local chatbot name: {name}")
        # If model is to be run remotely
        else:
            match model_config["remote"].lower():
                case "openai":
                    from .OpenAIChatbot import OpenAIChatbot
                    self.chatbot = OpenAIChatbot(model_config = model_config)
                # case "bard":
                #     from .BardChatbot import BardChatbot
                #     self.chatbot = BardChatbot()
                case _:
                    raise ValueError(f"Invalid remote chatbot name: {name}")

        self.name = self.chatbot.name
        self.keeps_context = self.chatbot.keeps_context #@ probably removing this

    def send_message(self,
                     message,
                     stop_sequences = [],
                     stop_regex = None,
                     n_tokens = 128):
        """
        Send a message to the chatbot and return the response. Wrapper around 
        automatically selected chatbot's send_message method.
        """

        return self.chatbot.send_message(message,
                                         stop_sequences,
                                         stop_regex,
                                         n_tokens)


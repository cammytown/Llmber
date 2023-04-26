from .Chatbot import Chatbot

class AutoChatbot(Chatbot):
    chatbot: Chatbot

    def __init__(self, name = "AutoChatbot", model_config: dict = { "name": "gpt2" }):
        super().__init__(name)

        # If model is to be run locally
        if "remote" not in model_config or not model_config["remote"]:
            match model_config["name"].lower():
                case "gpt2" | "pygmalion" | "decapoda-research/llama-7b-hf":
                    from .HFTransformersAutoBot import HFTransformersAutoBot
                    self.chatbot = HFTransformersAutoBot(name = model_config["name"])
                case "rwkv":
                    from .RWKVChatbot import RWKVChatbot
                    self.chatbot = RWKVChatbot()
                case "llamacpp":
                    from .LlamaCPPChatbot import LlamaCPPChatbot
                    self.chatbot = LlamaCPPChatbot()
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

        self.keeps_context = self.chatbot.keeps_context

    def send_message(self, message, stop_sequences = [], stop_regex = None):
        return self.chatbot.send_message(message, stop_sequences, stop_regex)


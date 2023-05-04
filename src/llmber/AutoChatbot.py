from .Chatbot import Chatbot

class AutoChatbot(Chatbot):
    """
    A chatbot that can be used to send messages to and receive messages from
    a LLM. Automatically selects chatbot based on model_config.
    """

    valid_options = ["name",
                     "remote",
                     "keep_context",
                     "keep_response_in_context"]
    chatbot: Chatbot

    def __init__(self,
                 name = "AutoChatbot",
                 logdir = "",
                 model_config: dict = { "name": "gpt2" }):

        super().__init__(name, model_config = model_config, logdir = logdir)

        # If model is to be run locally
        if "remote" not in model_config or not model_config["remote"]:
            match model_config["name"].lower():
                case "gpt2" | "pygmalion" | "decapoda-research/llama-7b-hf":
                    from .HFTAutoBot import HFTAutoBot
                    bot_class = HFTAutoBot
                case "rwkv":
                    from .RWKVChatbot import RWKVChatbot
                    bot_class = RWKVChatbot
                case "llamacpp":
                    from .LlamaCPPChatbot import LlamaCPPChatbot
                    bot_class = LlamaCPPChatbot
                case _:
                    raise ValueError(f"Invalid local chatbot name: {name}")

        # If model is to be run remotely
        else:
            match model_config["remote"].lower():
                case "openai":
                    from .OpenAIChatbot import OpenAIChatbot
                    bot_class = OpenAIChatbot
                # case "bard":
                #     from .BardChatbot import BardChatbot
                #     bot_class = BardChatbot()
                case _:
                    raise ValueError(f"Invalid remote chatbot name: {name}")

        self.chatbot = bot_class(model_config=model_config, logdir=logdir)

        #@REVISIT:
        self.name = self.chatbot.name
        self.keep_context = self.chatbot.keep_context
        self.keep_response_in_context = self.chatbot.keep_response_in_context

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


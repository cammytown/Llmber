from .chatbot import Chatbot

class AutoChatbot(Chatbot):
    """
    A chatbot that can be used to send messages to and receive messages from
    a LLM. Automatically selects chatbot based on model_config.
    """

    valid_options = ["engine",
                     "model",
                     "keep_context",
                     "keep_response_in_context"]
    chatbot: Chatbot

    def __init__(self,
                 model = "AutoChatbot",
                 logdir = "",
                 model_config: dict = {}):

        super().__init__(model,
                         model_config = model_config,
                         logdir = logdir)

        # Get engine and remove from model_config
        if "engine" not in model_config:
            # raise ValueError("AutoChatbot config requires 'engine'")
            model_config["engine"] = "hft_autobot"

        # if "model" not in model_config:
        #     raise ValueError("AutoChatbot config requires 'model'")

        engine = model_config["engine"].lower()
        model_config.pop("engine")

        # Determine engine
        match engine:
            case "huggingface" | "hft" | "transformers" | "hft_autobot":
                from .hft_autobot import HFTAutoBot
                bot_class = HFTAutoBot

            case "llamacpp":
                from .llamacpp_chatbot import LlamaCPPChatbot
                bot_class = LlamaCPPChatbot

            case "rwkv":
                from .rwkv_chatbot import RWKVChatbot
                bot_class = RWKVChatbot

            case "openai":
                from .openai_chatbot import OpenAIChatbot
                bot_class = OpenAIChatbot

            # case "bard":
            #     from .bard_chatbot import BardChatbot
            #     bot_class = BardChatbot()

            case _:
                raise ValueError(f"Invalid local chatbot engine: {engine}")

        self.chatbot = bot_class(model_config=model_config, logdir=logdir)

        #@REVISIT:
        self.model = self.chatbot.model
        self.keep_context = self.chatbot.keep_context
        self.keep_response_in_context = self.chatbot.keep_response_in_context

    def get_context(self):
        return self.chatbot.get_context()

    def set_context(self, context):
        return self.chatbot.set_context(context)

    def save_context(self):
        return self.chatbot.save_context()

    def restore_context(self):
        return self.chatbot.restore_context()

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


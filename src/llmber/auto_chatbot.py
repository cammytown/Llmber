from .chatbot import Chatbot

class AutoChatbot(Chatbot):
    """
    A chatbot that can be used to send messages to and receive messages from
    a LLM. Automatically selects chatbot based on model_config.
    The first argument can be either:
    - A string: interpreted as the model name
    - A dict: full model configuration
    """

    valid_options = ["engine",
                     "model",
                     "api_env_var",
                     "temperature",
                     "top_k",
                     "top_p",
                     "repeat_penalty",
                     "presence_penalty",
                     "keep_context",
                     "keep_response_in_context",
                     "use_cuda",
                     "max_context_length"]
    chatbot: Chatbot

    def __init__(self,
                 model_config: dict | str = {},
                 logdir = "",
                 ):

        # Convert string input to dict with model name
        if isinstance(model_config, str):
            model_config = {"model": model_config}

        super().__init__("AutoChatbot",
                         model_config = model_config,
                         logdir = logdir)

        # Get engine and remove from model_config
        if "engine" not in model_config:
            # raise ValueError("AutoChatbot config requires 'engine'")
            model_config["engine"] = "hft_autobot"

        # if "model" not in model_config:
        #     raise ValueError("AutoChatbot config requires 'model'")

        # Get engine
        engine = model_config["engine"].lower()

        # Remove engine from model_config
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
        # self.model = self.chatbot.model
        self.keep_context = self.chatbot.keep_context
        self.keep_response_in_context = self.chatbot.keep_response_in_context
        self.is_remote = self.chatbot.is_remote
        # self.use_cuda = self.chatbot.use_cuda

    def tokenize(self, string):
        return self.chatbot.tokenize(string)

    def detokenize(self, tokens):
        return self.chatbot.detokenize(tokens)

    def add_tokens_to_context(self, tokens):
        return self.chatbot.add_tokens_to_context(tokens)

    def add_string_to_context(self, string):
        return self.chatbot.add_string_to_context(string)

    def get_state(self):
        return self.chatbot.get_state()

    def set_state(self, state):
        return self.chatbot.set_state(state)

    def save_state(self):
        return self.chatbot.save_state()

    def restore_state(self):
        return self.chatbot.restore_state()

    def clear_state(self):
        return self.chatbot.clear_state()

    def sample(self,
               temperature = 0.8,
               top_k = 30,
               top_p = 0.95,
               repeat_penalty = 1.1,
               presence_penalty = 0.0):
        return self.chatbot.sample(temperature,
                                   top_k,
                                   top_p,
                                   repeat_penalty,
                                   presence_penalty)

    def send_message(self,
                     message,
                     stop_sequences = [],
                     n_tokens = 128):
        """
        Send a message to the chatbot and return the response. Wrapper around 
        automatically selected chatbot's send_message method.
        """

        return self.chatbot.send_message(message,
                                         stop_sequences,
                                         n_tokens)

    def generate_string(self, n_tokens=128, stop_sequences=[]):
        return self.chatbot.generate_string(n_tokens, stop_sequences)

    def generate_tokens(self, n_tokens=128, stop_sequences=[]):
        return self.chatbot.generate_tokens(n_tokens, stop_sequences)


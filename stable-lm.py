import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from Chatbot import Chatbot

class StableLMChatbot(Chatbot):
    def __init__(self, name = "StableLMChatbot"):
        super().__init__(name)

        model_path = "stabilityai/stablelm-tuned-alpha-7b"
        cache_dir = "/run/media/cammy/PROJECTS2/huggingface_cache"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, cache_dir=cache_dir)
        self.model.half().cuda()


    def send_message(self, message, stop_sequences = [], stop_regex = None):
        system_prompt = """<|SYSTEM|># StableLM Tuned (Alpha version)
        - StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
        - StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
        - StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
        - StableLM will refuse to participate in anything that could harm a human.
        """

        prompt = f"{system_prompt}<|USER|>What's your mood today?<|ASSISTANT|>"

        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        tokens = self.model.generate(
          **inputs,
          max_new_tokens=64,
          temperature=0.7,
          do_sample=True,
          stopping_criteria=StoppingCriteriaList([self.StopOnTokens()])
        )
        print(self.tokenizer.decode(tokens[0], skip_special_tokens=True))

    class StopOnTokens(StoppingCriteria):
        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
            stop_ids = [50278, 50279, 50277, 1, 0]
            for stop_id in stop_ids:
                if input_ids[0][-1] == stop_id:
                    return True
            return False

test = StableLMChatbot()
test.send_message("Hello")

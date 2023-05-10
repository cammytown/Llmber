from llmber.llamacpp_chatbot import LlamaCPPChatbot

chatbot = LlamaCPPChatbot()
response = chatbot.send_message("My favorite fruit is the delicious round red")
print(f"Response: {response}")


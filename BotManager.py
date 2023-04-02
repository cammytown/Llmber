import json
import os
import subprocess
from datetime import datetime
import asyncio

# from Chatbot import Chatbot
from BardChatbot import BardChatbot
from OpenAIChatbot import OpenAIChatbot
from AlpacaChatbot import AlpacaChatbot

from CoquiImp import CoquiImp

class BotManager:
    tts = CoquiImp()

    bots: Dict[str, Type[Chatbot]] = {}

    processing_requests = False
    request_queue = []

    conversation_mode_on = False
    data_dir = "chatbot-data"
    current_convo_filepath = f"{data_dir}/current-conversation.txt"
    convo_history_dir = f"{data_dir}/conversations"

    def __init__(self):
        print('bot init')

        # Load chatbots:
        bots["openai"] = OpenAIChatbot()
        bots["googlebard"] = BardChatbot()
        bots["alpaca"] = AlpacaChatbot()

        # Test TTS:
        self.sayFile('greeting.txt')

    def queue_request(self, bot_name, request):
        # Add request to array:
        self.request_queue.append({ 'bot_name': bot_name, 'request': request })

        # Start up the processing loop if it isn't going already:
        if(not self.processing_requests):
            asyncio.create_task(self.process_requests())

    async def process_requests(self):
        self.processing_requests = True

        while self.processing_requests:
            if len(self.request_queue) > 0:
                # process the oldest request in the queue
                request = self.request_queue.pop(0)
                print(f"Processing request for bot '{request['bot_name']}'"
                    + f"with request '{request['request']}'")

                self.process_request(request['bot_name'], request['request'])
            else:
                self.processing_requests = False
                # wait for a short time if the queue is empty
                #await asyncio.sleep(0.1)

    def process_request(self, bot_name, request):
        interaction = {
            "bot": bot_name,
            "request": request,
            "response": ""
        }

        #@REVISIT
        assert(bot_name != None)
        assert(request != None)
        print('bot_name: ' + bot_name)
        print('request: ' + request)

        # Create bot-specific directory if it doesn't exist:
        bot_directory = f"{data_dir}/{bot_name}"
        os.makedirs(bot_directory, exist_ok=True)

        # Make the request:
        response = self.ask_bot(bot_name, request)
        # Add interaction to global log:
        interaction['response'] = response
        with open(f"{bot_directory}/interaction-log.txt", "a") as history_file:
            history_file.write(json.dumps(interaction) + "\n")

        # Speak the response:
        self.say(interaction['response'])

    def activate_conversation_mode(self):
        # If there's a previous conversation, move it to history:
        if(os.path.exists(self.current_convo_filepath)):
            # Create the directory (if it doesn't already exist)
            if not os.path.exists(self.convo_history_dir):
                os.makedirs(self.convo_history_dir)

            # Get the current date and time as a formatted string
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            convo_filename = f"{self.convo_history_dir}/{timestamp}.txt"

            # Move and rename the previous conversation:
            os.rename(self.current_convo_filepath, convo_filename)

        # Set the property:
        self.conversation_mode_on = True

    def deactivate_conversation_mode(self):
        self.conversation_mode_on = False

    def ask_bot(self, bot_name, user_message) -> str:
        print(f"asking {bot_name}…")

        # If in conversation mode, prepend conversation history:
        #@REVISIT technically we could store the current conversation in a
        #class variable and just write to a file when the conversation ends; but
        #this is probably fine and keeps track even if a crash happens
        if(self.conversation_mode_on):
            # If a conversation has been started (a convo file exists):
            if(os.path.exists(self.current_convo_filepath)):
                # Open current convo file:
                with open(self.current_convo_filepath, "r") as convo_file:
                    # Load current convo:
                    current_conversation = convo_file.read()

                    # Prepare user message as dialogue script:
                    dialogue_entry = f"USER: {user_message}\nASSISTANT: "

                    # Prepend current convo to user_message:
                    user_message = current_conversation + dialogue_entry

        # Sent the request:
        bot_response = self.send_message_to_bot(bot_name, user_message)

        # If in conversation mode:
        if(self.conversation_mode_on):
            # Open current convo file:
            with open(self.current_convo_filepath, "a") as convo_file:
                # Compose new dialogue:
                interaction_script = f"USER: {user_message}\n"
                interaction_script += f"ASSISTANT: {bot_response}\n"

                # Append new dialogue to conversation file:
                convo_file.write(interaction_script)

        self.say(bot_response)

        # Return the response:
        return bot_response


    #@REVISIT naming:
    def send_message_to_bot(self, bot_name: str, user_message: str):
        # Select bot:
        chatbot = bots[bot_name]

        if chatbot == None:
            raise Exception(f"ERROR: unhandled bot_name {bot_name}")

        # Call bot:
        response = chatbot.send_message(user_message)

        return response




    def say(self, message):
        # Write response message to file mostly for festival tts:
        response_filepath = "chatbots/current-response.txt"
        with open(response_filepath, "w") as response_file:
            response_file.write(message)

        self.tts.say(message)
        # self.sayFile(response_filepath)

    def sayFile(self, file):
        print("Speaking greeting…")

        #@TODO obviously more complex handling:
        subprocess.Popen(["festival", "--tts", file])

from collections import Counter
import secretstorage

class Chatbot:
    # Setup secretstorage:
    _bus = secretstorage.dbus_init()
    _collection = secretstorage.get_default_collection(_bus)

    def __init__(self, name):
        self.name = name

    def retrieve_key(self, api_name):
        key_search = self._collection.search_items({'api': api_name})
        key = next(key_search)

        if key:
            secret = key.get_secret().decode('utf-8')
            return secret
        else:
            return False

    def send_message(self, user_message):
        response = openai.ChatCompletion.create(
          model="gpt-3.5-turbo",
          messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_message},
            ]
        )

        response_message = response['choices'][0]['message']['content']

        with open("chatbots/ChatGPT/request-usage.txt", "r+") as file:
            # Read file:
            previous_usage = Counter(json.load(file))

            # Combine previous usage with query usage:
            request_usage = Counter(response['usage'])
            total_usage = dict(previous_usage + request_usage)

            # Reset file read cursor to start:
            file.seek(0)

            # Write new total to file:
            json.dump(total_usage, file)

            # Remove file content after the cursor we just wrote up to:
            file.truncate();

        return responseMessage

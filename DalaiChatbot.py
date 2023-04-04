class DalaiChatbot(Chatbot):
    def __init__(self, name = "Dalai"):
        self.name = name

    def send_message(self, user_message):
        model = "/mnt/Files/.cache/dalai/alpaca/models/7B/ggml-model-q4_0.bin"
        #model = "/mnt/Files/.cache/dalai/alpaca/models/13B/ggml-model-q4_0.bin"

        alpaca = subprocess.Popen(["/mnt/Files/.cache/dalai/alpaca/main",
                                   "--seed", "-1",
                                   "--threads", "6",
                                   "--n_predict", "5",
                                   "--model", model,
                                   "--top_k", "40",
                                   "--top_p", "0.9",
                                   "--temp", "0.8",
                                   "--repeat_last_n", "64",
                                   "--repeat_penalty", "1.3",
                                   "-p", user_message
                                   ], stdout=subprocess.PIPE, text=True)
        #], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        response = None

        while True:
            # Predictions are output to stdout:
            output = alpaca.stdout.readline()

            if not output:
                break

            print('line:')
            print(output)

            # If this is the first line:
            if(response == None):
                # Remove the user's query from the returned text:
                response = output.replace(user_message, "")
            else:
                response += output

        return response

        # response = alpaca.communicate()  # wait for process to complete
        # print('response')
        # print(response)

        # while alpaca.poll() is None:
        #     stdout = alpaca.stdout.readline()
        #     if stdout:
        #         print(stdout.strip())
        #     else:
        #         time.sleep(0.1)

        # # Read remaining stdout after process completes
        # stdout, _ = alpaca.communicate()
        # if stdout:
        #     print(stdout.strip())

        # print("ALPACA DONE")


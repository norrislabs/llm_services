import sys
import time
import argparse
import pyfiglet

from llm_rest_client import ContextClient


if __name__ == "__main__":
    print(pyfiglet.figlet_format("LLM Test"))

    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--host", type=str, default="ai-001.local", help="server host")
    ap.add_argument("-p", "--port", type=int, default=8080, help="server port")
    args = vars(ap.parse_args())

    # Create the conversation REST client
    conv_client = ContextClient("test1", args["host"], args['port'])

    # Not needed. Just to have the time module used
    time.sleep(0.01)

    # Start a unique conversation with the LLM
    status = conv_client.create_context()
    print(status)

    # Create an error (dup conversation name)
#    status = conv_client.start_conversation()
#    print(status)

    # Get a list of conversations
    status = conv_client.get_context_names()
    print(status)

    # Prompt and get the response from the LLM
    # Method 1
    for word in conv_client.converse("how many moons does Pluto have?"):
        sys.stdout.write(word + " ")
        sys.stdout.flush()

    print()
    print()

    for word in conv_client.converse("and Saturn?"):
        sys.stdout.write(word + " ")
        sys.stdout.flush()

    # Method 2
#    predict_response = conv_client.prompt_only("how many moons does Pluto have?")
#    time.sleep(5.0)    # Do stuff
#    if predict_response.status_code == 200:
#        for word in conv_client.response_generator():
#            sys.stdout.write(word + " ")
#            sys.stdout.flush()

    print()

    # End the conversation
    end_response = conv_client.delete_context()
    print(end_response)

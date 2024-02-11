import os
import sys
import socket
import argparse
import pyfiglet
import click
import json
from colorama import Fore
import pprint
import pyperclip
import re

from llm_rest_client import ContextClient, LLMClient

current_context = ""
contexts = {}
context_colors = {}

conv_colors = [Fore.GREEN, Fore.CYAN, Fore.MAGENTA, Fore.LIGHTGREEN_EX,
               Fore.LIGHTBLUE_EX, Fore.WHITE, Fore.RED, Fore.LIGHTRED_EX]


def start_context(context_name, host, port, history=2):
    global contexts
    global context_colors
    global current_context

    # Create a new context client
    con_client = ContextClient(context_name, host, port)
    status = con_client.create_context(history=history, system_prompt="")
    if not status[0]:
        print(status[1])
        return False, None

    # Assign the next color to the conversation for its response
    contexts[context_name] = con_client
    next_color_index = len(contexts) % len(conv_colors)
    context_colors[context_name] = conv_colors[next_color_index - 1]

    # Switch to this new context
    current_context = context_name
    return True, con_client


def clear_screen():
    sys.stderr.write("\x1b[2J\x1b[H")


def filter_words(theword):
    filter_these = ['response:', 'ai:', 'answer:']
    return any(theword.lower().strip().startswith(w) for w in filter_these)


def print_info(msg, indent=0):
    print(" " * indent + Fore.LIGHTBLACK_EX + msg + Fore.RESET)


def print_warning(msg, indent=0):
    print(" " * indent + Fore.LIGHTYELLOW_EX + msg + Fore.RESET)


def print_error(msg, indent=0):
    print(" " * indent + Fore.LIGHTRED_EX + msg + Fore.RESET)


def load_questions(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
        return lines
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return []


def show_help():
    try:
        with open("help.txt", 'r') as file:
            lines = file.readlines()

        print()
        print("{} Command Help {}".format('*' * 25, '*' * 40))
        for line in lines:
            print("  " + line.strip())
        print()
    except FileNotFoundError:
        print_error(f"Help file not found.")


def display_model_info():
    # Display LLM model information
    global args
    print("Current model information:")
    info = llm_client.get_model_info()[2]
    info['host'] = args['host'] + ':' + str(args['port'])
    print(json.dumps(info, indent=4))
    print()


def display_context_info(cname):
    print("Current information for context '{}':".format(cname))
    info = contexts[cname].get_context_info()
    print(json.dumps(info, indent=4))
    print()


if __name__ == "__main__":
    print(pyfiglet.figlet_format("LLM Client"))

    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--host", type=str, default="ai-001.local", help="server host")
    ap.add_argument("-p", "--port", type=int, default=8080, help="server port")
    ap.add_argument("-q", "--questions", type=str, default="science_questions_100.txt",
                    help="Questions file path")
    args = vars(ap.parse_args())

    # Preload questions
    questions = load_questions(args['questions'])
    question_start = 0
    question_index = question_start
    question_end = len(questions) - 1
    auto_question = False

    llm_client = LLMClient(args['host'], args['port'])

    # Display LLM model information
    display_model_info()

    # List current existing contexts
    names_status = llm_client.get_context_names()
    if names_status[0]:
        names = sorted(names_status[2])
        if len(names) > 0:
            print("The following contexts are active:")
            for name in names:
                print_info(name, 2)
            print()
    else:
        print_error("Error {} getting names of contexts.".format(names_status[1]))

    # Build a default context name
    user_login_id = os.getlogin().lower()
    host_name = socket.gethostname().lower()
    default_context_name = "{}-{}".format(user_login_id, host_name.split('.')[0])

    # Get a context name from the user
    current_context = click.prompt("Enter a new or existing context name: ", type=click.types.STRING,
                                   default=default_context_name)

    # Now create a default context
    stat = start_context(current_context, args['host'], args['port'])
    if not stat[0]:
        print("Unable to create context '{}'".format(current_context))
        sys.exit(0)

    human_msg = ""
    while True:
        if not auto_question:
            try:
                human_msg = input("{}({}):{} ".format(context_colors[current_context],
                                                      current_context,
                                                      Fore.RESET)).strip()
            except EOFError:
                print(Fore.RESET + "Goodbye and good luck.")
                break

        if len(human_msg) > 0:
            if human_msg == ".quit" or human_msg == ".q":
                print("So long and thanks for all the fish.")
                break

            # Clear the screen
            if human_msg == ".cls":
                clear_screen()
                continue

            # Display model info
            if human_msg.startswith(".info"):
                cmd_items = human_msg.split()
                if len(cmd_items) == 2:
                    if cmd_items[1] == "model":
                        display_model_info()
                        continue
                    elif cmd_items[1] == "context":
                        display_context_info(current_context)
                        continue
                print("Invalid command.")
                continue

            if human_msg == ".help":
                show_help()
                continue

            # Restart LLM
            if human_msg == ".restart":
                llm_client.restart_llm()
                contexts = {}
                context_colors = {}

                # Create the 'default' context
                start_context(default_context_name, args['host'], args['port'])
                print("My mind is clear now.")
                continue

            # Get message from clipboard
            if human_msg == ".paste":
                human_msg = pyperclip.paste().strip()
                print("Text from clipboard:")
                print_info(human_msg)
                print()

            # Create/Start a new context
            if human_msg.startswith(".start "):
                cmd_items = human_msg.split()
                if len(cmd_items) > 1:
                    if cmd_items[1] not in contexts:
                        if len(cmd_items) == 2:
                            start_context(cmd_items[1], args['host'], args['port'])
                            print_warning("Created context '{}'.".format(cmd_items[1]))
                        elif len(cmd_items) == 3:
                            print(cmd_items)
                            start_context(cmd_items[1], args['host'], args['port'], int(cmd_items[2]))
                            print_warning("Created context '{}' with history of {}.".format(cmd_items[1],
                                                                                            cmd_items[2]))
                        else:
                            print_error("Invalid context command.")
                    else:
                        print_warning("Context '{}' already exists.".format(cmd_items[1]))
                else:
                    print_error("Invalid context command.")
                continue

            # Switch contexts
            if human_msg.startswith(".switch ") or human_msg.startswith(".sw "):
                cmd_items = human_msg.split()
                if len(cmd_items) != 2:
                    print_error("Invalid command.")
                elif cmd_items[1] in contexts:
                    start_context(cmd_items[1], args['host'], args['port'])
                    print_warning("Switched to context '{}'.".format(cmd_items[1]))
                    last_loaded_template = ""
                else:
                    print_error("Context '{}' does not exist.".format(cmd_items[1]))
                continue

            # Erase current context's history
            if human_msg == ".forget":
                contexts[current_context].clear_context()
                print("My mind is going. I can feel it.")
                continue

            # Display current context's history
            if human_msg == ".history":
                hist = contexts[current_context].get_history()
                if hist:
                    for item in hist:
                        pprint.pprint(item, sort_dicts=False)
                else:
                    print_warning("Unable to get any history.")
                continue

            # Set the current context's system prompt
            if human_msg.startswith(".prompt"):
                cmd_items = human_msg.split()
                if len(cmd_items) > 1:
                    sentence = ' '.join(cmd_items[1:])
                    resp = contexts[current_context].set_system_prompt(sentence.strip())
                    print_info("System prompt set to '{}'.".format(sentence))
                else:
                    resp = contexts[current_context].set_system_prompt("")
                    print_info("Set empty system prompt.")
                continue

            # Delete a context
            if human_msg.startswith(".del "):
                cmd_items = human_msg.split()
                if len(cmd_items) != 2:
                    print_error("Invalid delete context command.")
                elif cmd_items[1] in contexts:
                    if cmd_items[1] != current_context:
                        contexts[cmd_items[1]].delete_context()
                        del contexts[cmd_items[1]]
                        del context_colors[cmd_items[1]]
                        print_info("Deleted context '{}'.".format(cmd_items[1]))
                    else:
                        print_warning("You cannot delete the current context.".format(cmd_items[1]))
                else:
                    print_error("Context '{}' does not exist.".format(cmd_items[1]))
                continue

            # Get/load prompt template for current context
            if human_msg.startswith(".template"):
                cmd_items = human_msg.split()
                if len(cmd_items) >= 2:
                    if cmd_items[1].strip().lower() == "show":
                        templ = contexts[current_context].get_template()
                        if templ[1] == 200:
                            tfile, remainder = templ[2]['detail'].split('|', 1)
                            if len(remainder.strip()) == 0:
                                remainder = "Template has not be rendered yet."
                            print_info("Loaded from '{}'\n {}".format(tfile, remainder), 2)
                        else:
                            print_warning(templ[2]['detail'], 2)
                        continue

                    elif cmd_items[1].strip().lower() == "load":
                        if len(cmd_items) == 3:
                            templ_file = cmd_items[2]
                        else:
                            if contexts[current_context].last_loaded_template:
                                templ_file = contexts[current_context].last_loaded_template
                            else:
                                print_warning("Load a template first.", 2)
                                continue

                        templ = contexts[current_context].load_template(templ_file)
                        if templ[1] == 200:
                            last_loaded_template = templ_file
                            print_info("Loaded template '{}'.".format(templ_file), 2)
                        else:
                            print_warning("Unable to load template '{}'.".format(templ_file), 2)
                        continue

                    print_error("Invalid template subcommand '{}'.".format(cmd_items[1]))
                continue

            # List all current contexts
            if human_msg == ".list":
                names_status = llm_client.get_context_names()
                if names_status[0]:
                    for name in sorted(names_status[2]):
                        print_info(name, 2)
                else:
                    print_error("Error {} getting names of contexts.".format(names_status[1]))
                continue

            # Get the next question from the loaded question file
            if auto_question or human_msg == ".ask":
                human_msg = questions[question_index].strip()
                print("{}. {}".format(question_index + 1, human_msg))
                question_index += 1
                if question_index > question_end:
                    question_index = question_start
                    auto_question = False

            # Select a question number or set a question auto range
            if human_msg.startswith(".ask "):
                items = human_msg.split(" ")

                if len(items) == 2:
                    question_index = int(items[1]) - 1
                    human_msg = questions[question_index].strip()
                    print("{}. {}".format(question_index + 1, human_msg))
                    question_index += 1

                elif len(items) == 3:
                    question_index = int(items[1]) - 1
                    question_end = int(items[2]) - 1
                    human_msg = questions[question_index].strip()
                    print("{}. {}".format(question_index + 1, human_msg))
                    question_index += 1
                    auto_question = True

                else:
                    print_error("Invalid question command.")
                    continue

            # Test questions
            if human_msg == ".1":
                human_msg = "How many moons does pluto have?"
            elif human_msg == ".2":
                human_msg = "and saturn?"
            elif human_msg == ".3":
                human_msg = "why is pluto no longer a planet?"
            elif human_msg == ".4":
                human_msg = "the IAU should be defunded for what it did to poor Pluto"

            # Send a prompt to the LLM and stream the response
            max_line_len = 120
            current_line_len = 0
            pattern = r'(.*?)[.: ]\d+\.'

            try:
                resp_gen, resp_id = contexts[current_context].submit_directive(human_msg)
                for word in resp_gen:
                    if not filter_words(word):
                        wd = word.replace("\n", "")

                        # Split numbered lines
                        match = re.match(pattern, wd)
                        if match is not None:
                            wd = wd[0:len(match.group(1))+1] + "\n" + wd[len(match.group(1))+1:]
                            current_line_len = len(match.group(1))+1

                        # Word wrap
                        current_line_len += len(wd) + 1
                        if current_line_len > max_line_len:
                            print('\n', end='', flush=True)
                            current_line_len = 0

                        print(context_colors[current_context] + wd + " ", end="", flush=True)
                print(Fore.RESET)

            except ValueError as ex:
                print_error(str(ex))

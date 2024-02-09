import time
import queue
import logging
import threading
import uuid
from typing import Union
from abc import ABC, abstractmethod
from pydantic import BaseModel
import importlib

from llm_streamers import Word2StdoutStreamer, Word2QueueStreamer
from llm_contexts import Context


class Directive(BaseModel):
    response_id: str
    context_name: str
    msg: str


class BaseLanguageModel(ABC):
    streamer_types = {'stdout': Word2StdoutStreamer,
                      'queue': Word2QueueStreamer,
                      'none': None}

    def __init__(self, model, template, verbose=False, **kwargs):
        self._llm, self._model_info = self.create_llm(model, verbose, kwargs)
        self._default_template_file = template

        self._contexts = {}
        self._directive_queue = queue.Queue()
        self._directives_thread = threading.Thread(target=self._directive_worker, args=())

        self._running = True
        self._directives_thread.start()
        self._last_result = None

    @abstractmethod
    def create_llm(self, model, verbose, kwargs):
        pass

    def __enter__(self):
        pass

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        self.shutdown()

    def shutdown(self):
        # Shutdown the converse worker thread
        if self._running:
            self._running = False
            self._directives_thread.join()

        # Delete all contexts
        for context_name in list(self._contexts):
            self.delete_context(context_name)

        self._contexts = {}
        self._directive_queue = queue.Queue()

        logging.info("LLM has been shutdown.")

    def restart(self):
        self.shutdown()

        self._directives_thread = threading.Thread(target=self._directive_worker, args=())
        self._running = True
        self._directives_thread.start()

        logging.info("LLM has been restarted.")

    def _directive_worker(self):
        while self._running:
            try:
                directive = self._directive_queue.get(block=False)

                if directive.context_name in self._contexts:
                    logging.info("Started directive in context '{}'...".format(directive.context_name))

                    # Send directive message to the selected context
                    # This blocks here while working
                    self._last_result = self._contexts[directive.context_name].predict(directive.response_id,
                                                                                       directive.msg)

                    logging.info("Completed directive in context '{}'\n\n".format(directive.context_name))
                else:
                    logging.error("Unknown context '{}'".format(directive.context_name))

            except queue.Empty:
                time.sleep(0.1)

    def create_context(self, context_name: str,
                       template_file: Union[str, None] = None,
                       history_count: int = 2,
                       system_prompt: Union[str, None] = None,
                       summerizer_type: str = "abstractive",
                       streamer_type: Union[str, None] = None, **streamer_params) -> bool:
        if context_name not in self._contexts:
            # Select prompt template (specified here or specified with the LLM)
            templ_file = template_file if template_file else self._default_template_file
            if not templ_file:
                logging.error("Context '{}' does not have a template.".format(context_name))
                return False

            # Setup a streamer
            streamer = None
            if streamer_type in self.streamer_types:
                streamer = self.streamer_types[streamer_type](context_name, **streamer_params)

            # Create context class based on the name located in the first line of the template file
            try:
                with open('templates/' + templ_file, 'r') as file:
                    # Read the first line from the file
                    template_type = file.readline().strip()
            except FileNotFoundError:
                logging.warning("Template file '{}' not found.".format(template_file))
                return False

            try:
                # Import the module dynamically
                module = importlib.import_module("llm_contexts")
                conv = getattr(module, template_type)(context_name, self._llm, templ_file, history_count,
                                                      system_prompt, summerizer_type, streamer)
            except AttributeError:
                logging.error("Context type '{}' does not exist.".format(template_type))
                return False

            self._contexts[context_name] = conv
            logging.info("Started {} context '{}' with history of {}.".format(template_type, context_name,
                                                                              history_count))
            return True
        else:
            logging.warning("Reusing context '{}'.".format(context_name))
            return False

    def delete_context(self, context_name: str) -> bool:
        if context_name in self._contexts:
            logging.info("Ended context '{}'.".format(context_name))
            self._contexts.pop(context_name)
            return True
        else:
            logging.error("Unknown context '{}'.".format(context_name))
            return False

    def clear_context(self, context_name: str) -> bool:
        if context_name in self._contexts:
            self._contexts[context_name].erase_memory()
            return True
        else:
            logging.error("Unknown context '{}'.".format(context_name))
            return False

    def submit_directive(self, context_name: str, msg: str) -> Union[str, None]:
        # Add new human message to the LLM queue
        if context_name in self._contexts:
            resp_id_full = uuid.uuid4().hex
            resp_id = resp_id_full[0:4] + resp_id_full[-4:]
            directive_item = Directive(response_id=resp_id, context_name=context_name, msg=msg)
            self._directive_queue.put(directive_item)
            return directive_item.response_id
        else:
            logging.error("Unknown context '{}'.".format(context_name))
            return None

    def load_template(self, context_name: str, template_file: str) -> bool:
        if context_name in self._contexts:
            return self._contexts[context_name].load_template(template_file)

    def get_template(self, context_name: str) -> Union[str, None]:
        if context_name in self._contexts:
            return self._contexts[context_name].template_file + "|" + self._contexts[context_name].template
        else:
            logging.error("Unknown context '{}'.".format(context_name))
            return None

    def set_system_prompt(self, context_name: str, system_prompt: str) -> bool:
        if context_name in self._contexts:
            self._contexts[context_name].system_prompt = system_prompt
            return True
        else:
            logging.error("Unknown context '{}'.".format(context_name))
            return False

    def get_context(self, context_name: str) -> Union[Context, None]:
        if context_name in self._contexts:
            return self._contexts[context_name]
        else:
            logging.error("Unknown context '{}'.".format(context_name))
            return None

    def get_history(self, context_name: str):
        if context_name in self._contexts:
            return self._contexts[context_name].history
        else:
            logging.error("Unknown context '{}'.".format(context_name))
            return None

    def get_context_names(self):
        names = list(self._contexts.keys())
        return names

    @property
    def model_info(self):
        return self._model_info

    @property
    def last_result(self):
        return self._last_result

import sys
from queue import Queue
from typing import Any, Dict, List
import demoji
import logging

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult


class PromptCallbackHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> Any:
        formatted_prompts = "\n".join(prompts)
        logging.info(f"Prompt:\n\n{formatted_prompts}")


class Word2StdoutStreamer(BaseCallbackHandler):
    """Callback handler used to handle callbacks from langchain and output to stdout."""

    def __init__(self, name, **params):
        super(BaseCallbackHandler, self).__init__()
        self._con_id = "???"
        self._name = name
        self._word = ""
        self._token_count = 0
        self._params = params

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        """Run when chain starts running."""
        sys.stdout.write("|START-{}|".format(self._name))
        sys.stdout.flush()

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when chain ends running."""
        word = self._filter_word(self._word)
        sys.stdout.write(word + "|END-{}|\n".format(self._token_count))
        sys.stdout.flush()
        self._word = ""
        self._token_count = 0

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        self._token_count += 1
        if token.startswith(' '):
            word = self._filter_word(self._word)

            if word.startswith("*") and word.endswith("*"):
                word = '<' + word[1:-1] + ">"
            else:
                if word.startswith("*"):
                    word = '<' + word[1:]
                elif word.endswith("*"):
                    word = word[0:-1] + ">"

            sys.stdout.write(word + " ")
            sys.stdout.flush()
            self._word = token
        else:
            self._word += token

    @property
    def id(self):
        return self._con_id

    @id.setter
    def id(self, value):
        self._con_id = value

    @staticmethod
    def _filter_word(word):
        # Remove AI prefix and emojis
        wd = word.strip().replace("AI:", "")
        return demoji.replace(wd, "")


class Word2QueueStreamer(BaseCallbackHandler):
    """Callback handler used to handle callbacks from langchain and output to a queue."""

    def __init__(self, name, **params):
        super(BaseCallbackHandler, self).__init__()
        self._name = name
        self._word = ""
        self._token_count = 0
        self._params = params
        self._streamer_queue = Queue()
        self._resp_id = "???"

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        """Run when chain starts running."""
#        print("Starting {}".format(self._resp_id))
        start_code = "|START-{}-{}|".format(self._resp_id, self._name)
        self._streamer_queue.put(start_code)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run when chain ends running."""
        word = self._filter_word(self._word)
        self._streamer_queue.put(word)
        end_code = "|END-{}-{}-{}|".format(self._resp_id, self._name, self._token_count)
        self._streamer_queue.put(end_code)
        self._word = ""
        self._token_count = 0

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        self._token_count += 1
        if token.startswith(' '):
            word = self._filter_word(self._word)

            if word.startswith("*") and word.endswith("*"):
                word = '<' + word[1:-1] + ">"
            else:
                if word.startswith("*"):
                    word = '<' + word[1:]
                elif word.endswith("*"):
                    word = word[0:-1] + ">"

            self._streamer_queue.put(word)
            self._word = token
        else:
            self._word += token

    def get_word(self):
        return self._streamer_queue.get()

    def processed_word(self):
        self._streamer_queue.task_done()

    def no_words(self):
        return self._streamer_queue.empty()

    @property
    def id(self):
        return self._resp_id

    @id.setter
    def id(self, value):
        self._resp_id = value

    @staticmethod
    def _filter_word(word):
        # Remove damn AI generated emojis
        wd = word.strip()
        return demoji.replace(wd, "")

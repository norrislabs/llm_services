import logging
from typing import Union
from jinja2 import Environment, FileSystemLoader, exceptions
from abc import ABC, abstractmethod

import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from collections import Counter
from heapq import nlargest
from transformers import pipeline

from langchain.callbacks.base import BaseCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, ConversationChain
from langchain.memory import ConversationBufferWindowMemory

from llm_streamers import PromptCallbackHandler


class Context(ABC):

    @abstractmethod
    def __init__(self, name, llm, template_file: str, history_count: int,
                 system_prompt: Union[str, None] = None,
                 summerizer_type: str = 'abstractive',
                 streamer: Union[BaseCallbackHandler, None] = None):
        self._name = name
        self._llm = llm
        self._template_file = template_file
        self._history_count = history_count * 2
        self._out_streamer = streamer
        self._system_prompt = system_prompt

        self._history = []

        # Set up the history summerizer
        self._nlp = None
        self._summarizer = None
        self._summerizer_type = summerizer_type

        if summerizer_type == 'abstractive':
            self._summarizer = pipeline("summarization")

        elif summerizer_type == 'extractive':
            self._nlp = spacy.load('en_core_web_sm')

        elif summerizer_type == 'none':
            pass

        else:
            logging.error("Unknown summerizer type '{}'".format(summerizer_type))
            self._summerizer_type = "none"

        self._template_text = ""

    @abstractmethod
    def erase_memory(self):
        pass

    @abstractmethod
    def predict(self, stream_id, message):
        pass

    @abstractmethod
    def load_template(self, template_file) -> bool:
        pass

    def summerize(self, text: str) -> str:
        if self._summerizer_type == 'abstractive':
            return self.summerize_abstractive(text)

        elif self._summerizer_type == 'extractive':
            return self.summerize_extractive(text)

        return text

    def summerize_extractive(self, text: str) -> str:
        doc = self._nlp(text)

        keyword = []
        stopwords = list(STOP_WORDS)
        pos_tag = ['PROPN', 'ADJ', 'NOUN', 'VERB']
        for token in doc:
            if token.text in stopwords or token.text in punctuation:
                continue
            if token.pos_ in pos_tag:
                keyword.append(token.text)

        freq_word = Counter(keyword)

        max_freq = Counter(keyword).most_common(1)[0][1]
        for word in freq_word.keys():
            freq_word[word] = int(freq_word[word]/max_freq)
        freq_word.most_common(5)

        sent_strength = {}
        for sent in doc.sents:
            for word in sent:
                if word.text in freq_word.keys():
                    if sent in sent_strength.keys():
                        sent_strength[sent] += freq_word[word.text]
                    else:
                        sent_strength[sent] = freq_word[word.text]

        summarized_sentences = nlargest(3, sent_strength, key=sent_strength.get)
        final_sentences = [w.text for w in summarized_sentences]
        summary = ' '.join(final_sentences)
        return summary

    def summerize_abstractive(self, text: str) -> str:
        text_sum = self._summarizer(text, min_length=10, max_length=80)
        summary = ' '.join([i['summary_text'] for i in text_sum])
        return summary

    @property
    def system_prompt(self):
        return self._system_prompt

    @system_prompt.setter
    def system_prompt(self, value):
        logging.info("Set system prompt in context '{}'".format(self._name))
        self._system_prompt = value

    @property
    def template(self):
        return self._template_text

    @property
    def template_file(self):
        return self._template_file

    @property
    def streamer(self) -> BaseCallbackHandler:
        return self._out_streamer

    @property
    def history(self):
        return self._history


class ContextInstruct(Context):
    def __init__(self, name, llm, template_file: str, history_count: int,
                 system_prompt: Union[str, None] = None,
                 summerizer_type: str = 'abstractive',
                 streamer: Union[BaseCallbackHandler, None] = None):
        super().__init__(name, llm, template_file, history_count, system_prompt, summerizer_type, streamer)

        self._j_template = None
        self.load_template(template_file)

    def erase_memory(self):
        self._history = []

    def predict(self, stream_id, message):
        # Build messages list from message and history
        messages = []

        # Add system prompt if specified
        if len(self._system_prompt) > 0:
            messages.append({"role": "system", "content": self._system_prompt})

        # TODO
        # Add example user/assistant dialog from somewhere

        # Add history of user/assistant interactions
        for hist in self._history:
            messages.append(hist)

        # Add user query/directive
        query = {'role': 'user', 'content': message}
        messages.append(query)
        messages.append({"role": "assistant", "content": ""})

        # Use the above messages to render prompt template using jinja2
        self._template_text = self._j_template.render(messages=messages)
        prompt = PromptTemplate.from_template(self._template_text)
        chain = LLMChain(llm=self._llm, prompt=prompt, verbose=False)

        # Invoke the LLM!
        if self._out_streamer:
            # Wait while streaming the output
            self._out_streamer.id = stream_id
            result = chain.invoke({}, {"callbacks": [self._out_streamer,
                                                     PromptCallbackHandler()]})
        else:
            # Wait quietly for the full result
            result = chain.invoke({})

        # Save query/response to history
        summerized = ""
        if self._history_count > 0:
            if self._history_count == len(self._history):
                self._history.pop(0)
                self._history.pop(0)

            # Save query to history
            self._history.append(query)

            # Summerize the response before saving it to history
            summerized = self.summerize(result['text'].strip())
            self._history.append({'role': 'assistant', 'content': summerized})

        return result['text'].strip(), summerized

    def load_template(self, template_file) -> bool:
        try:
            j_environ = Environment(loader=FileSystemLoader("templates/"))
            self._j_template = j_environ.get_template(template_file)
            logging.info("Loaded template '{}' into context {}".format(template_file, self._name))
            self._template_text = ""    # not rendered yet
            self._template_file = template_file
            return True
        except exceptions.TemplateNotFound:
            logging.warning("Template file '{}' not found.".format(template_file))
            return False


class ContextStandard(Context):
    def __init__(self, name, llm, template_file: str, history_count: int,
                 system_prompt: Union[str, None] = None,
                 summerizer_type: str = 'extractive',
                 streamer: Union[BaseCallbackHandler, None] = None):
        super().__init__(name, llm, template_file, history_count, system_prompt, summerizer_type, streamer)

        self._prompt = None
        self._input_vars = ["input", "history"]
        self.load_template(template_file)

        # Setup history
        if self._history_count > 0:
            self._chat_memory = ConversationBufferWindowMemory(k=self._history_count)
        else:
            self._chat_memory = None

    def erase_memory(self):
        self._chat_memory.clear()

    def predict(self, stream_id, prompt):
        ccc = ConversationChain(
            llm=self._llm,
            prompt=self._prompt,
            memory=self._chat_memory,
            verbose=True)

        # Invoke the LLM!
        if self._out_streamer:
            # Wait while streaming the output
            self._out_streamer.id = stream_id
            result = ccc.predict([self._out_streamer, PromptCallbackHandler()], input=prompt)
        else:
            # Wait quietly for the full result
            result = ccc.predict(input=prompt)

        summerized = ""
        if self._history_count > 0:
            if self._history_count == len(self._history):
                self._history.pop(0)
                self._history.pop(0)

            # Save query to history
            query = {'role': 'user', 'content': prompt}
            self._history.append(query)

            # Summerize the response and save it to history
            summerized = self.summerize(result.strip())
            self._history.append({'role': 'assistant', 'content': summerized})

        return result.strip(), summerized

    def load_template(self, template_file) -> bool:
        try:
            with open('templates/' + template_file, 'r') as f:
                self._template_text = f.read()
                self._prompt = PromptTemplate(input_variables=self._input_vars,
                                              template=self._template_text)
                logging.info("Loaded standard template '{}' into context {}".format(template_file,
                                                                                    self._name))
                self._template_file = template_file
                return True
        except FileNotFoundError:
            logging.warning("Template file '{}' not found.".format(template_file))
            return False

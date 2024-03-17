import requests
import re


class LLMClient:
    def __init__(self, host, port, prefix="http"):
        self._llm_url = "{}://{}:{}/llm/".format(prefix, host, port)

    @staticmethod
    def _build_return_status(resp):
        return resp.status_code == 200 or resp.status_code == 422, resp.status_code, resp.json()

    def restart_llm(self):
        resp = requests.post(self._llm_url + "restart")
        return self._build_return_status(resp)

    def shutdown_llm(self):
        resp = requests.post(self._llm_url + "shutdown")
        return self._build_return_status(resp)

    # Get a list of all the active conversation names
    def get_context_names(self):
        resp = requests.get(self._llm_url + "list")
        return self._build_return_status(resp)

    # Get a list of all the active conversation names
    def get_model_info(self):
        resp = requests.get(self._llm_url + "info")
        return self._build_return_status(resp)


class ContextClient:
    def __init__(self, conv_name, host, port, prefix="http"):
        self._name = conv_name
        self._con_url = "{}://{}:{}/context/".format(prefix, host, port)
        self._current_respid = ""
        self._last_loaded_template = ""

    @staticmethod
    def _build_return_status(resp):
        return resp.status_code == 200 or resp.status_code == 422, resp.status_code, resp.json()

    # Start a named context
    def create_context(self, template="", history=2, system_prompt="", summerizer_type="abstractive"):
        try:
            json = {"template": template, "history": history,
                    "system_prompt": system_prompt, "summerizer_type": summerizer_type}
            resp = requests.post(self._con_url + self._name, json=json)
            if resp.status_code == 200:
                self._last_loaded_template = template
                tresp = self.get_template()
                if not tresp:
                    return False, 422, {}

        except requests.exceptions.ConnectionError:
            return False, 422, {}
        else:
            return self._build_return_status(resp)

    # End and delete a context and its history
    def delete_context(self):
        resp = requests.delete(self._con_url + self._name)
        return self._build_return_status(resp)

    # Erase a context's history
    def clear_context(self):
        resp = requests.patch(self._con_url + "history/" + self._name)
        return self._build_return_status(resp)

    # Get a context's history
    def get_history(self):
        resp = requests.get(self._con_url + "history/" + self._name)
        if resp.status_code == 200:
            return resp.json()['detail']
        else:
            return None

    # Get a context's information
    def get_context_info(self):
        resp = requests.get(self._con_url + "info/" + self._name)
        if resp.status_code == 200:
            return resp.json()['detail']
        else:
            return None

    # Set context's system prompt
    def set_system_prompt(self, prompt: str):
        json = {"system_prompt": prompt}
        resp = requests.put(self._con_url + "prompt/" + self._name, json=json)
        return self._build_return_status(resp)

    # Load a context's prompt template
    def load_template(self, template: str):
        json = {"template": template}
        resp = requests.put(self._con_url + "template/" + self._name, json=json)
        if resp.status_code == 200:
            self._last_loaded_template = template
        return self._build_return_status(resp)

    # Get a context's current rendered template
    def get_template(self):
        resp = requests.get(self._con_url + "template/" + self._name)
        result = self._build_return_status(resp)
        if resp.status_code == 200:
            ttext: str = result[2]['detail']
            self._last_loaded_template, _ = ttext.split('|', 1)
        return result

    # Send directive (prompt) to the context and return a response generator
    def submit_directive(self, msg: str):
        json = {"msg": msg}
        resp = requests.put(self._con_url + self._name, json=json)
        if resp.status_code == 200:
            self._current_respid = resp.json()['detail']
            return self.response_generator(), self._current_respid
        else:
            return None, '|ERROR-{}, {}, {}|'.format(resp.status_code, resp.reason, resp.text)

    # Just send directive and return
    # Used with response_generator to do stuff between the directive and getting a response
    def directive_only(self, msg: str):
        json = {"human_msg": msg}
        resp = requests.put(self._con_url + "converse/" + self._name, json=json)
        self._current_respid = resp.json()['detail']
        return self._build_return_status(resp)

    @staticmethod
    def _match_marker(string):
        pattern = r'^\|[^|]*-+[^|]*\|$'
        return re.match(pattern, string) is not None

    def _is_resp_marker(self, word: str):
        if self._match_marker(word):
            return True, word[1:len(word) - 1].split("-")
        else:
            return False, word

    # Response generator
    # Only used directly in conjuction with prompt_only
    def response_generator(self):
        # sending a request and fetching a response which is stored in r
        with requests.get(self._con_url + self._name, stream=True) as r:
            is_recv = False
            for chunk in r.iter_content(128):
                word = chunk.decode("utf-8")

                is_marker, data = self._is_resp_marker(word)
                if is_marker and self._current_respid == data[1]:
                    if data[0] == 'START':
                        assert not is_recv, "Got a start inside of a response."
                        is_recv = True
                    elif data[0] == 'END':
                        assert is_recv, "Got a end before the start of a response."
                        is_recv = False
                    continue

                if is_recv:
                    yield word

    @property
    def last_loaded_template(self):
        return self._last_loaded_template

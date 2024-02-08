from jinja2 import Environment, FileSystemLoader

messages = [{'role': 'system', 'content': 'This is the system prompt.'},

            {'role': 'user', 'content': 'This is the first user input.'},
            {'role': 'assistant', 'content': 'This is the first assistant response.'},
            {'role': 'user', 'content': 'This is the second user input.'},
            {'role': 'assistant', 'content': 'This is the second assistant response.'},

            {'role': 'user', 'content': 'how many moons does Pluto have?'},
            {"role": "assistant", "content": "x"}
            ]

messages2 = [{'role': 'user', 'content': 'How many moons does pluto have?'},
             {'role:': 'assistant', 'content': 'As of now, Pluto is known to have five officially recognized moons: Charon, Nix, Hydra, and Kerberos. These were discovered between the late 1970s and the early 2000s using ground-based telescopes and spacecraft like New Horizons.\n\nThere are also several unconfirmed candidates for moons of Pluto that have been detected by various surveys, but they require further confirmation through follow-up observations or dedicated missions to the Pluto system.'},
             {'role': 'user', 'content': 'and saturn?'}, {'role': 'assistant', 'content': ''}]

environment = Environment(loader=FileSystemLoader("templates/"))
template = environment.get_template("mistral.jinja2")

print(template.render(messages=messages))

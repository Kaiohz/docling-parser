import os


class PromptLoader:
    directory = "prompts"

    def __init__(self):
        self.prompts = self._load_prompts()

    def _load_prompts(self):
        prompts = {}
        for filename in os.listdir(self.directory):
            if filename.endswith(".prompt"):
                name = os.path.splitext(filename)[0]
                with open(os.path.join(self.directory, filename), "r") as file:
                    content = file.read()
                    prompts[name] = content
        return prompts


prompts = PromptLoader().prompts

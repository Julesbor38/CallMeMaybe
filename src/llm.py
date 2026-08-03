from llm_sdk import Small_LLM_Model


class LLM:
    def __init__(self, model_name: str):
        self.model = Small_LLM_Model(model_name)

    def encode(self, text: str):
        return self.model.encode(text)

    def decode(self, ids):
        return self.model.decode(ids)

    def logits(self, ids):
        return self.model.get_logits_from_input_ids(ids)

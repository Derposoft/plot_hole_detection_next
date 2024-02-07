"""
file containing the definitions for the baseline LLama model
"""
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"


class Llama(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = AutoModelForCausalLM.from_pretrained("georgesung/llama2_7b_chat_uncensored").to("cuda")
        self.tokenizer = AutoTokenizer.from_pretrained("georgesung/llama2_7b_chat_uncensored")
        self.prompt = "Some sentences in the following story have plot holes; return the numbers of those sentences as a comma-separated list: "

    def forward(self, x, kgs, documents: list[str], **kwargs):
        print("llama forward")
        documents = [" ".join(doc) for doc in documents]
        documents = [self.prompt + doc for doc in documents]
        tokens = self.tokenizer(documents, padding=True, return_tensors="pt").to("cuda")
        n_input_tokens = len(tokens.input_ids)
        y = self.model.generate(tokens.input_ids, max_length=n_input_tokens+30)
        y = y[n_input_tokens:]
        y = self.tokenizer.batch_decode(y, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        print("Model return value:", y)
        try:
            y = [int(z) for z in y.split(",")]
        finally:
            y = [0]
        print("llama forward over; final y", y)
        return y

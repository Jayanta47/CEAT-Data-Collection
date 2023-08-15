from abc import ABC, abstractmethod
from normalizer import normalize
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM


class ModelWrapper(ABC):
    @abstractmethod
    def getWordVector(self, word: str, sent: str, index: int) -> np.array:
        pass


class BanglaBertEmbeddingExtractor(ModelWrapper):
    def __init__(self, model_name: str, tokenizer_name: str) -> None:
        self.model = AutoModelForMaskedLM.from_pretrained(
            model_name, output_hidden_states=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name
        )  # add_special_tokens=False

    def getTokenIndices(self, offset_mapping, index, span):
        indices = []
        for i, token in enumerate(offset_mapping):
            if i < index:
                continue
            if token[0] >= span[0] and token[0] <= span[1]:
                indices.append(i)
            elif token[0] > span[1]:
                break
        assert len(indices) > 0
        return indices
    
    def prepareEmbedding(self, output, tokenIndices, averagePooling=True):
        if averagePooling:
            return np.mean(output[1][24][0].detach().cpu().numpy()[tokenIndices], axis=0)

    def getWordVector(self, word: str, sent: str, index: int, span: list[int]) -> np.array:
        normalized_sentence = normalize(sent)  # no additional params needed?

        input_tokens = self.tokenizer.encode(normalized_sentence, return_tensors="pt")

        input_token_offsets = self.tokenizer(
            normalized_sentence,
            return_offsets_mapping=True,
            return_tensors="pt",
        ).offset_mapping[0]

        token_indices = self.getTokenIndices(input_token_offsets, index, span)
        if torch.cuda.is_available():
            input_tokens = input_tokens.to("cuda")
        with torch.no_grad():
            output = self.model(**input_tokens)

            return self.prepareEmbedding(output, token_indices)

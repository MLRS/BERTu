from overrides import overrides
from typing import Any, Dict, List

import torch
import torch.nn as nn

from allennlp.data.vocabulary import Vocabulary
from allennlp.models.heads import Head
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from allennlp.nn.initializers import InitializerApplicator
from allennlp.training.metrics.f1_measure import FBetaMeasure
from torch.nn import CrossEntropyLoss


@Head.register("linear_classifier")
class LinearTagger(Head):
    def __init__(
        self,
        vocab: Vocabulary,
        encoder: Seq2SeqEncoder,
        dropout: float = 0.0,
        initializer: InitializerApplicator = InitializerApplicator(),
        **kwargs,
    ) -> None:
        super().__init__(vocab, **kwargs)

        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(
            encoder.get_output_dim(), vocab.get_vocab_size("sentiment")
        )

        self.f1 = FBetaMeasure(average="macro")

        initializer(self)

    @overrides
    def forward(
        self,
        embedded_text_input: torch.Tensor,
        mask: torch.Tensor,
        sentiment: torch.Tensor = None,
        metadata: List[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        embedded_text_input = self.dropout(embedded_text_input[:, -1])
        encoded_text = self.encoder(embedded_text_input)

        batch = self.dropout(encoded_text)
        logits = self.linear(batch)

        outputs = {"logits": logits, "mask": mask}

        if sentiment is not None:
            loss = CrossEntropyLoss()(logits, sentiment)
            self.f1(logits, sentiment)

            outputs["loss"] = loss
        if metadata is not None:
            outputs["metadata"] = metadata
        return outputs

    @overrides
    def make_output_human_readable(
        self,
        output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the probabilities, converts index to string label, and
        add `"label"` key to the dictionary with the result.
        """
        predictions = torch.nn.functional.softmax(output_dict["logits"], dim=-1)
        if predictions.dim() == 2:
            predictions = [predictions[i] for i in range(predictions.shape[0])]
        else:
            predictions = [predictions]

        def decode_label(label):
            return str(label)

        output_dict["labels"] = [decode_label(prediction.argmax(dim=-1).item()) for prediction in predictions]
        return output_dict

    @overrides
    def get_metrics(self, reset: bool) -> Dict[str, float]:
        return {
            **self.f1.get_metric(reset),
        }

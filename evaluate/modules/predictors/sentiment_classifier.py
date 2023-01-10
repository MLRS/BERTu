from allennlp.common.util import JsonDict
from allennlp.predictors.predictor import Predictor
from allennlp.predictors.text_classifier import TextClassifierPredictor


@Predictor.register("sentiment_classifier")
class SentimentClassifier(TextClassifierPredictor):

    def dump_line(self, outputs: JsonDict) -> str:
        sentence = outputs["sentiment_metadata"]["words"]
        label = outputs["sentiment_labels"]

        return f"{label},{sentence}"

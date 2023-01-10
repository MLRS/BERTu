from allennlp.common.util import JsonDict
from allennlp.predictors.predictor import Predictor
from allennlp.predictors.sentence_tagger import SentenceTaggerPredictor


@Predictor.register("named_entity_recogniser")
class NamedEntityRecogniser(SentenceTaggerPredictor):

    def dump_line(self, outputs: JsonDict) -> str:
        tokens = outputs["ner_words"]
        tags = outputs["ner_tags"]

        assert len(tokens) == len(tags)
        return "\n".join(f"mt:{tokens[i]}\t{tags[i]}" for i in range(len(tokens))) + "\n\n"

from allennlp.common.util import JsonDict
from allennlp.predictors.predictor import Predictor
from allennlp.predictors.sentence_tagger import SentenceTaggerPredictor


@Predictor.register("part_of_speech_tagger")
class PartOfSpeechPredictor(SentenceTaggerPredictor):

    def dump_line(self, outputs: JsonDict) -> str:
        tokens = outputs["pos_words"]
        tags = outputs["pos_tags"]

        assert len(tokens) == len(tags)
        return "\n".join(f"{i}\t{tokens[i]}\t{tags[i]}" for i in range(len(tokens))) + "\n\n"

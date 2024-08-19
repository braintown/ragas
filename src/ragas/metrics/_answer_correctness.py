from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass, field

import numpy as np
from langchain_core.pydantic_v1 import BaseModel

from ragas.llms.output_parser import RagasoutputParser, get_json_format_instructions
from ragas.llms.prompt import Prompt, PromptValue
from ragas.metrics._answer_similarity import AnswerSimilarity
from ragas.metrics._faithfulness import (
    LONG_FORM_ANSWER_STATEMENTS_PROMPT,
    HasSegmentMethod,
    _statements_output_parser,
)
from ragas.metrics.base import (
    EvaluationMode,
    MetricWithEmbeddings,
    MetricWithLLM,
    get_segmenter,
)
from ragas.run_config import RunConfig

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks


logger = logging.getLogger(__name__)


class AnswerCorrectnessClassification(BaseModel):
    TP: t.List[t.Dict[str, t.Any]]
    FP: t.List[t.Dict[str, t.Any]]
    FN: t.List[t.Dict[str, t.Any]]


_output_instructions = get_json_format_instructions(AnswerCorrectnessClassification)
_output_parser = RagasoutputParser(pydantic_object=AnswerCorrectnessClassification)

CORRECTNESS_INSTRUCTIONS = """\
Given a ground truth and an answer statements, analyze each statement and classify them in one of the following categories:

- TP (true positive): statements that are present in answer that are also directly supported by the one or more statements in ground truth,
- FP (false positive): statements present in the answer but not directly supported by any statement in ground truth,
- FN (false negative): statements found in the ground truth but not present in answer.

Each statement can only belong to one of the categories. Provide a reason for each classification.
"""

CORRECTNESS_INSTRUCTIONS_NEW = """\
给定一个地面事实和一个答案陈述，分析每个陈述并将其分类为以下类别之一：

- TP（真正的正例）：在答案中存在的陈述也直接得到地面事实中的一个或多个陈述的支持，
- FP（错误的正例）：答案中存在的陈述但没有直接得到地面事实中的任何陈述的支持，
- FN（漏报的负例）：在地面事实中找到但在答案中不存在的陈述。

每个陈述只能属于其中一个类别。请为每个分类提供理由。
"""

CORRECTNESS_PROMPT = Prompt(
    name="answer_correctness",
    instruction=CORRECTNESS_INSTRUCTIONS,
    output_format_instruction=_output_instructions,
    examples=[
        {
            "question": """What powers the sun and what is its primary function?""",
            "answer": [
                "The sun is powered by nuclear fission, similar to nuclear reactors on Earth.",
                "The primary function of the sun is to provide light to the solar system.",
            ],
            "ground_truth": [
                "The sun is powered by nuclear fusion, where hydrogen atoms fuse to form helium.",
                "This fusion process in the sun's core releases a tremendous amount of energy.",
                "The energy from the sun provides heat and light, which are essential for life on Earth.",
                "The sun's light plays a critical role in Earth's climate system.",
                "Sunlight helps to drive the weather and ocean currents.",
            ],
            "classification": AnswerCorrectnessClassification.parse_obj(
                {
                    "TP": [
                        {
                            "statement": "The primary function of the sun is to provide light to the solar system.",
                            "reason": "This statement is somewhat supported by the ground truth mentioning the sun providing light and its roles, though it focuses more broadly on the sun's energy.",
                        }
                    ],
                    "FP": [
                        {
                            "statement": "The sun is powered by nuclear fission, similar to nuclear reactors on Earth.",
                            "reason": "This statement is incorrect and contradicts the ground truth which states that the sun is powered by nuclear fusion.",
                        }
                    ],
                    "FN": [
                        {
                            "statement": "The sun is powered by nuclear fusion, where hydrogen atoms fuse to form helium.",
                            "reason": "This accurate description of the sun’s power source is not included in the answer.",
                        },
                        {
                            "statement": "This fusion process in the sun's core releases a tremendous amount of energy.",
                            "reason": "This process and its significance are not mentioned in the answer.",
                        },
                        {
                            "statement": "The energy from the sun provides heat and light, which are essential for life on Earth.",
                            "reason": "The answer only mentions light, omitting the essential aspects of heat and its necessity for life, which the ground truth covers.",
                        },
                        {
                            "statement": "The sun's light plays a critical role in Earth's climate system.",
                            "reason": "This broader impact of the sun’s light on Earth's climate system is not addressed in the answer.",
                        },
                        {
                            "statement": "Sunlight helps to drive the weather and ocean currents.",
                            "reason": "The effect of sunlight on weather patterns and ocean currents is omitted in the answer.",
                        },
                    ],
                }
            ).dict(),
        },
        {
            "question": """What is the boiling point of water?""",
            "answer": [
                "The boiling point of water is 100 degrees Celsius at sea level"
            ],
            "ground_truth": [
                "The boiling point of water is 100 degrees Celsius (212 degrees Fahrenheit) at sea level.",
                "The boiling point of water can change with altitude.",
            ],
            "classification": AnswerCorrectnessClassification.parse_obj(
                {
                    "TP": [
                        {
                            "statement": "The boiling point of water is 100 degrees Celsius at sea level",
                            "reason": "This statement is directly supported by the ground truth which specifies the boiling point of water as 100 degrees Celsius at sea level.",
                        }
                    ],
                    "FP": [],
                    "FN": [
                        {
                            "statement": "The boiling point of water can change with altitude.",
                            "reason": "This additional information about how the boiling point of water can vary with altitude is not mentioned in the answer.",
                        }
                    ],
                }
            ).dict(),
        },
    ],
    input_keys=["question", "answer", "ground_truth"],
    output_key="classification",
    output_type="json",
)


CORRECTNESS_PROMPT_NEW = Prompt(
    name="answer_correctness_new",
    instruction=CORRECTNESS_INSTRUCTIONS_NEW,
    output_format_instruction=_output_instructions,
    examples=[
        {
            "question": """是什么为太阳提供能量，它的主要功能是什么？""",
            "answer": [
                "太阳由核裂变提供能量，类似于地球上的核反应堆。",
                "太阳的主要功能是为太阳系提供光。",
            ],
            "ground_truth": [
                "太阳由核聚变提供能量，其中氢原子聚变形成氦。",
                "太阳核心的这种聚变过程释放出大量的能量。",
                "来自太阳的能量提供了热量和光，这对地球上的生命至关重要。",
                "太阳的光在地球的气候系统中起着关键作用。",
                "阳光有助于驱动天气和海洋洋流。",
            ],
            "classification": AnswerCorrectnessClassification.parse_obj(
                {
                    "TP": [
                        {
                            "statement": "太阳的主要功能是为太阳系提供光。",
                            "reason": "这一陈述在一定程度上得到了地面事实的支持，地面事实提到了太阳提供光及其作用，尽管它更广泛地关注太阳的能量。",
                        }
                    ],
                    "FP": [
                        {
                            "statement": "太阳由核裂变提供能量，类似于地球上的核反应堆。",
                            "reason": "这一陈述是不正确的，与地面事实相矛盾，地面事实指出太阳由核聚变提供能量。",
                        }
                    ],
                    "FN": [
                        {
                            "statement": "太阳由核聚变提供能量，其中氢原子聚变形成氦。",
                            "reason": "这一对太阳能量来源的准确描述没有包含在答案中。",
                        },
                        {
                            "statement": "太阳核心的这种聚变过程释放出大量的能量。",
                            "reason": "这一过程及其重要性在答案中没有提到。",
                        },
                        {
                            "statement": "来自太阳的能量提供了热量和光，这对地球上的生命至关重要。",
                            "reason": "答案中只提到了光，而省略了热量及其对生命的必要性，这在地面事实中有所涵盖。",
                        },
                        {
                            "statement": "太阳的光在地球的气候系统中起着关键作用。",
                            "reason": "太阳光对地球气候系统的广泛影响在答案中没有涉及。",
                        },
                        {
                            "statement": "阳光有助于驱动天气和海洋洋流。",
                            "reason": "答案中省略了阳光对天气模式和海洋洋流的影响。",
                        },
                    ],
                }
            ).dict(),
        },
        {
            "question": """水的沸点是多少？""",
            "answer": [
                "水的沸点是100摄氏度，在海平面上。"
            ],
            "ground_truth": [
                "水的沸点是100摄氏度（212华氏度），在海平面上。",
                "水的沸点会随海拔高度变化。",
            ],
            "classification": AnswerCorrectnessClassification.parse_obj(
                {
                    "TP": [
                        {
                            "statement": "水的沸点是100摄氏度，在海平面上。",
                            "reason": "这一陈述直接得到了地面事实的支持，地面事实明确指出水的沸点在海平面上是100摄氏度。",
                        }
                    ],
                    "FP": [],
                    "FN": [
                        {
                            "statement": "水的沸点会随海拔高度变化。",
                            "reason": "这一关于水的沸点会随海拔高度变化的额外信息在答案中没有提到。",
                        }
                    ],
                }
            ).dict(),
        },
    ],
    input_keys=["question", "answer", "ground_truth"],
    output_key="classification",
    output_type="json",
)

@dataclass
class AnswerCorrectness(MetricWithLLM, MetricWithEmbeddings):
    """
    Measures answer correctness compared to ground truth as a combination of
    factuality and semantic similarity.

    Attributes
    ----------
    name: string
        The name of the metrics
    weights:
        a list of two weights corresponding to factuality and semantic similarity
        Defaults [0.75, 0.25]
    answer_similarity:
        The AnswerSimilarity object
    """

    name: str = "answer_correctness"  # type: ignore[reportIncompatibleMethodOverride]
    evaluation_mode: EvaluationMode = EvaluationMode.qga  # type: ignore[reportIncompatibleMethodOverride]
    correctness_prompt: Prompt = field(default_factory=lambda: CORRECTNESS_PROMPT_NEW)
    long_form_answer_prompt: Prompt = field(
        default_factory=lambda: LONG_FORM_ANSWER_STATEMENTS_PROMPT
    )
    weights: list[float] = field(default_factory=lambda: [0.75, 0.25])
    answer_similarity: t.Optional[AnswerSimilarity] = None
    sentence_segmenter: t.Optional[HasSegmentMethod] = None
    max_retries: int = 1

    def __post_init__(self: t.Self):
        if len(self.weights) != 2:
            raise ValueError(
                "Expects a list of two weights. First for factuality, second for semantic similarity"
            )
        if all([w == 0 for w in self.weights]):
            raise ValueError("At least one weight must be non-zero")
        if not all([w >= 0 for w in self.weights]):
            raise ValueError("Weights must be non-negative")

        if self.sentence_segmenter is None:
            language = self.long_form_answer_prompt.language
            self.sentence_segmenter = get_segmenter(language=language, clean=False)

    def init(self, run_config: RunConfig):
        super().init(run_config)
        if self.answer_similarity is None and self.weights[1] != 0:
            self.answer_similarity = AnswerSimilarity(
                llm=self.llm, embeddings=self.embeddings
            )

    def _compute_statement_presence(
        self, prediction: AnswerCorrectnessClassification
    ) -> float:
        tp = len(prediction.TP)
        fp = len(prediction.FP)
        fn = len(prediction.FN)
        score = tp / (tp + 0.5 * (fp + fn)) if tp > 0 else 0
        return score

    def _create_statements_prompt(self, question: str, text: str) -> PromptValue:
        assert self.sentence_segmenter is not None, "sentence_segmenter is not set"

        sentences = self.sentence_segmenter.segment(text)
        sentences = [
            sentence for sentence in sentences if sentence.strip().endswith(".")
        ]
        sentences = "\n".join([f"{i}:{x}" for i, x in enumerate(sentences)])
        prompt_value = self.long_form_answer_prompt.format(
            question=question, answer=text, sentences=sentences
        )
        return prompt_value

    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> t.Dict:
        assert self.llm is not None, "LLM must be set"

        question = row["question"]
        statements = {}
        for item in ["answer", "ground_truth"]:
            p_value = self._create_statements_prompt(question, row[item])
            item_statement = await self.llm.generate(p_value, callbacks=callbacks)
            statements[item] = await _statements_output_parser.aparse(
                item_statement.generations[0][0].text,
                p_value,
                self.llm,
                self.max_retries,
            )
            statements[item] = (
                statements[item].dicts() if statements[item] is not None else []
            )
        global answer
        if not all([val == [] for val in statements.values()]):
            ground_truth = [
                statement
                for item in statements["ground_truth"]
                for statement in item["simpler_statements"]
            ]
            answer = [
                statement
                for item in statements["answer"]
                for statement in item["simpler_statements"]
            ]

            p_value = self.correctness_prompt.format(
                question=question,
                ground_truth=ground_truth,
                answer=answer,
            )
            is_statement_present = await self.llm.generate(p_value, callbacks=callbacks)
            result_text = is_statement_present.generations[0][0].text

            answers = await _output_parser.aparse(
                result_text, p_value, self.llm, self.max_retries
            )
            if answers is None:
                return np.nan

            f1_score = self._compute_statement_presence(answers)
        else:
            f1_score = 1.0

        if self.weights[1] == 0:
            similarity_score = 0.0
        else:
            assert self.answer_similarity is not None, "AnswerSimilarity must be set"

            similarity_score = await self.answer_similarity.ascore(
                row, callbacks=callbacks
            )

        score = np.average(
            [f1_score, similarity_score],
            weights=self.weights,
        )

        return {"answer_correctness_list":answer,"scores":score }

    def adapt(self, language: str, cache_dir: t.Optional[str] = None) -> None:
        assert self.llm is not None, "llm must be set to compute score"

        logger.info(f"Adapting AnswerCorrectness metric to {language}")
        self.correctness_prompt = self.correctness_prompt.adapt(
            language, self.llm, cache_dir
        )

        self.sentence_segmenter = get_segmenter(language=language, clean=False)

    def save(self, cache_dir: t.Optional[str] = None) -> None:
        self.correctness_prompt.save(cache_dir)


answer_correctness = AnswerCorrectness()

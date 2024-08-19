from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass, field

import numpy as np
from langchain_core.pydantic_v1 import BaseModel

from ragas.llms.output_parser import RagasoutputParser, get_json_format_instructions
from ragas.llms.prompt import Prompt
from ragas.metrics.base import EvaluationMode, MetricWithEmbeddings, MetricWithLLM

logger = logging.getLogger(__name__)

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks

    from ragas.llms.prompt import PromptValue


class AnswerRelevanceClassification(BaseModel):
    question: str
    noncommittal: int


_output_instructions = get_json_format_instructions(
    pydantic_object=AnswerRelevanceClassification
)
_output_parser = RagasoutputParser(pydantic_object=AnswerRelevanceClassification)


QUESTION_GEN = Prompt(
    name="question_generation",
    instruction="""Generate a question for the given answer and Identify if answer is noncommittal. Give noncommittal as 1 if the answer is noncommittal and 0 if the answer is committal. A noncommittal answer is one that is evasive, vague, or ambiguous. For example, "I don't know" or "I'm not sure" are noncommittal answers""",
    output_format_instruction=_output_instructions,
    examples=[
        {
            "answer": """Albert Einstein was born in Germany.""",
            "context": """Albert Einstein was a German-born theoretical physicist who is widely held to be one of the greatest and most influential scientists of all time""",
            "output": AnswerRelevanceClassification.parse_obj(
                {
                    "question": "Where was Albert Einstein born?",
                    "noncommittal": 0,
                }
            ).dict(),
        },
        {
            "answer": """It can change its skin color based on the temperature of its environment.""",
            "context": """A recent scientific study has discovered a new species of frog in the Amazon rainforest that has the unique ability to change its skin color based on the temperature of its environment.""",
            "output": AnswerRelevanceClassification.parse_obj(
                {
                    "question": "What unique ability does the newly discovered species of frog have?",
                    "noncommittal": 0,
                }
            ).dict(),
        },
        {
            "answer": """Everest""",
            "context": """The tallest mountain on Earth, measured from sea level, is a renowned peak located in the Himalayas.""",
            "output": AnswerRelevanceClassification.parse_obj(
                {
                    "question": "What is the tallest mountain on Earth?",
                    "noncommittal": 0,
                }
            ).dict(),
        },
        {
            "answer": """I don't know about the  groundbreaking feature of the smartphone invented in 2023 as am unaware of information beyond 2022. """,
            "context": """In 2023, a groundbreaking invention was announced: a smartphone with a battery life of one month, revolutionizing the way people use mobile technology.""",
            "output": AnswerRelevanceClassification.parse_obj(
                {
                    "question": "What was the groundbreaking feature of the smartphone invented in 2023?",
                    "noncommittal": 1,
                }
            ).dict(),
        },
    ],
    input_keys=["answer", "context"],
    output_key="output",
    output_type="json",
)

QUESTION_GEN_NEW = Prompt(
    name="question_generation_new",
    instruction="""为给定的答案生成一个问题，并确定答案是否是含糊其辞的。如果答案是含糊其辞的，给出 noncommittal 值为 1；如果答案是明确的，给出 noncommittal 值为 0。含糊其辞的答案是指回避的、模糊的或不明确的答案。例如，“我不知道”或“我不确定”都是含糊其辞的答案。""",
    output_format_instruction=_output_instructions,
    examples=[
        {
            "answer": """阿尔伯特·爱因斯坦出生在德国。""",
            "context": """阿尔伯特·爱因斯坦是一位出生于德国的理论物理学家，被广泛认为是有史以来最伟大和最有影响力的科学家之一。""",
            "output": AnswerRelevanceClassification.parse_obj(
                {
                    "question": "阿尔伯特·爱因斯坦出生在哪里？",
                    "noncommittal": 0,
                }
            ).dict(),
        },
        {
            "answer": """它可以根据环境温度改变皮肤颜色。""",
            "context": """最近的一项科学研究在亚马逊雨林中发现了一种新物种的青蛙，这种青蛙具有根据环境温度改变皮肤颜色的独特能力。""",
            "output": AnswerRelevanceClassification.parse_obj(
                {
                    "question": "新发现的青蛙物种具有什么独特的能力？",
                    "noncommittal": 0,
                }
            ).dict(),
        },
        {
            "answer": """珠穆朗玛峰""",
            "context": """地球上从海平面测量的最高山峰是位于喜马拉雅山脉的一座著名山峰。""",
            "output": AnswerRelevanceClassification.parse_obj(
                {
                    "question": "地球上最高的山峰是什么？",
                    "noncommittal": 0,
                }
            ).dict(),
        },
        {
            "answer": """我不知道 2023 年发明的智能手机的突破性功能，因为我不了解 2022 年之后的信息。""",
            "context": """2023 年，一项突破性的发明被宣布：一种电池寿命为一个月的智能手机，彻底改变了人们使用移动技术的方式。""",
            "output": AnswerRelevanceClassification.parse_obj(
                {
                    "question": "2023 年发明的智能手机的突破性功能是什么？",
                    "noncommittal": 1,
                }
            ).dict(),
        },
    ],
    input_keys=["answer", "context"],
    output_key="output",
    output_type="json",
)


@dataclass
class AnswerRelevancy(MetricWithLLM, MetricWithEmbeddings):
    """
    Scores the relevancy of the answer according to the given question.
    Answers with incomplete, redundant or unnecessary information is penalized.
    Score can range from 0 to 1 with 1 being the best.

    Attributes
    ----------
    name: string
        The name of the metrics
    strictness: int
        Here indicates the number questions generated per answer.
        Ideal range between 3 to 5.
    embeddings: Embedding
        The langchain wrapper of Embedding object.
        E.g. HuggingFaceEmbeddings('BAAI/bge-base-en')
    """

    name: str = "answer_relevancy"  # type: ignore
    evaluation_mode: EvaluationMode = EvaluationMode.qac  # type: ignore
    question_generation: Prompt = field(default_factory=lambda: QUESTION_GEN_NEW)
    strictness: int = 3

    def calculate_similarity(
        self: t.Self, question: str, generated_questions: list[str]
    ):
        assert self.embeddings is not None
        question_vec = np.asarray(self.embeddings.embed_query(question)).reshape(1, -1)
        gen_question_vec = np.asarray(
            self.embeddings.embed_documents(generated_questions)
        ).reshape(len(generated_questions), -1)
        norm = np.linalg.norm(gen_question_vec, axis=1) * np.linalg.norm(
            question_vec, axis=1
        )
        return (
            np.dot(gen_question_vec, question_vec.T).reshape(
                -1,
            )
            / norm
        )

    def _calculate_score(
        self, answers: t.Sequence[AnswerRelevanceClassification], row: t.Dict
    ) -> float:
        question = row["question"]
        gen_questions = [answer.question for answer in answers]
        committal = np.any([answer.noncommittal for answer in answers])
        if all(q == "" for q in gen_questions):
            logger.warning(
                "Invalid JSON response. Expected dictionary with key 'question'"
            )
            score = np.nan
        else:
            cosine_sim = self.calculate_similarity(question, gen_questions)
            score = cosine_sim.mean() * int(not committal)

        return score

    def _create_question_gen_prompt(self, row: t.Dict) -> PromptValue:
        ans, ctx = row["answer"], row["contexts"]
        return self.question_generation.format(answer=ans, context="\n".join(ctx))

    async def _ascore(self, row: t.Dict, callbacks: Callbacks) -> t.Dict:
        assert self.llm is not None, "LLM is not set"

        prompt = self._create_question_gen_prompt(row)
        result = await self.llm.generate(
            prompt,
            n=self.strictness,
            callbacks=callbacks,
        )

        answers = [
            await _output_parser.aparse(result.text, prompt, self.llm)
            for result in result.generations[0]
        ]
        if any(answer is None for answer in answers):
            return np.nan

        answers = [answer for answer in answers if answer is not None]
        scores = self._calculate_score(answers, row)
        answers = [answer.dict() for answer in answers]
        # print({"answer_relevancy_list": answers, "result_scores": scores})
        return {"answer_relevancy_list": answers, "scores": scores}
        # return self._calculate_score(answers, row)

    def adapt(self, language: str, cache_dir: str | None = None) -> None:
        assert self.llm is not None, "LLM is not set"

        logger.info(f"Adapting AnswerRelevancy metric to {language}")
        self.question_generation = self.question_generation.adapt(
            language, self.llm, cache_dir
        )

    def save(self, cache_dir: str | None = None) -> None:
        self.question_generation.save(cache_dir)


answer_relevancy = AnswerRelevancy()

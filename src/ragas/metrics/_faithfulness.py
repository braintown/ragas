from __future__ import annotations

import json
import logging
import typing as t
from dataclasses import dataclass, field

import numpy as np
from langchain_core.pydantic_v1 import BaseModel, Field

from ragas.llms.output_parser import RagasoutputParser, get_json_format_instructions
from ragas.llms.prompt import Prompt
from ragas.metrics.base import EvaluationMode, MetricWithLLM, ensembler, get_segmenter

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks

    from ragas.llms.prompt import PromptValue

from typing import Any, Protocol


class HasSegmentMethod(Protocol):
    def segment(self, text) -> Any:
        ...


logger = logging.getLogger(__name__)


class Statements(BaseModel):
    sentence_index: int = Field(
        ..., description="Index of the sentence from the statement list"
    )
    simpler_statements: t.List[str] = Field(..., description="the simpler statements")


class StatementsAnswers(BaseModel):
    __root__: t.List[Statements]

    def dicts(self) -> t.List[t.Dict]:
        return self.dict()["__root__"]


_statements_output_instructions = get_json_format_instructions(StatementsAnswers)
_statements_output_parser = RagasoutputParser(pydantic_object=StatementsAnswers)


LONG_FORM_ANSWER_PROMPT = Prompt(
    name="long_form_answer",
    output_format_instruction=_statements_output_instructions,
    instruction="Given a question, an answer, and sentences from the answer analyze the complexity of each sentence given under 'sentences' and break down each sentence into one or more fully understandable statements while also ensuring no pronouns are used in each statement. Format the outputs in JSON.",
    examples=[
        {
            "question": "Who was Albert Einstein and what is he best known for?",
            "answer": "He was a German-born theoretical physicist, widely acknowledged to be one of the greatest and most influential physicists of all time. He was best known for developing the theory of relativity, he also made important contributions to the development of the theory of quantum mechanics.",
            "sentences": """
        0:He was a German-born theoretical physicist, widely acknowledged to be one of the greatest and most influential physicists of all time. 
        1:He was best known for developing the theory of relativity, he also made important contributions to the development of the theory of quantum mechanics.
        """,
            "analysis": StatementsAnswers.parse_obj(
                [
                    {
                        "sentence_index": 0,
                        "simpler_statements": [
                            "Albert Einstein was a German-born theoretical physicist.",
                            "Albert Einstein is recognized as one of the greatest and most influential physicists of all time.",
                        ],
                    },
                    {
                        "sentence_index": 1,
                        "simpler_statements": [
                            "Albert Einstein was best known for developing the theory of relativity.",
                            "Albert Einstein also made important contributions to the development of the theory of quantum mechanics.",
                        ],
                    },
                ]
            ).dicts(),
        }
    ],
    input_keys=["question", "answer", "sentences"],
    output_key="analysis",
    language="english",
)
LONG_FORM_ANSWER_STATEMENTS_PROMPT = Prompt(
    name="long_form_answer_statements",
    output_format_instruction=_statements_output_instructions,
    instruction="给定一个问题、一个答案和答案中的句子，分析“句子”下的每个句子的复杂性，并将每个句子分解为一个或多个完全可理解的陈述，同时确保每个陈述中不使用代词。之后，确定陈述中是否指出了客观问题，如果它们被指示为主观问候等，则删除该部分陈述。以 JSON 格式输出。",
    examples=[
        {
            "question": "Aura 的加热和通风座椅在长途驾驶中效果如何？",
            "answer": "IQ Aura 的加热和通风座椅非常棒，无论是寒冷的冬天还是炎热的夏天，都能保持座椅在舒适的温度。它在长途驾驶中尤其有效，让你一路舒适。你对驾驶体验的其他方面还有什么疑虑吗？",
            "sentences": """
        0:IQ Aura 的加热和通风座椅非常棒
        1:在寒冷的冬天和炎热的夏天都能保持座椅在舒适的温度
        2:它在长途驾驶中尤其有效，让你一路舒适
        3:你对驾驶体验的其他方面还有什么疑虑吗？
        """,
            "analysis": StatementsAnswers.parse_obj(
                [
                    {
                        "sentence_index": 0,
                        "simpler_statements": [
                            "IQ Aura 的加热座椅非常棒",
                            "IQ Aura 的通风座椅非常棒",
                        ],
                    },
                    {
                        "sentence_index": 1,
                        "simpler_statements": [
                            "IQ Aura 在寒冷的冬天保持座椅在舒适的温度",
                            "IQ Aura 在炎热的夏天保持座椅在舒适的温度",
                        ],
                    },
                    {
                        "sentence_index": 2,
                        "simpler_statements": [
                            "IQ Aura 在长途驾驶中尤其有效",
                            "IQ Aura 让你一路舒适",
                        ],
                    },
                ]
            ).dicts(),
        }
    ],
    input_keys=["question", "answer", "sentences"],
    output_key="analysis",
    language="chinese",
)


class StatementFaithfulnessAnswer(BaseModel):
    statement: str = Field(..., description="the original statement, word-by-word")
    reason: str = Field(..., description="the reason of the verdict")
    verdict: int = Field(..., description="the verdict(0/1) of the faithfulness.")


class StatementFaithfulnessAnswers(BaseModel):
    __root__: t.List[StatementFaithfulnessAnswer]

    def dicts(self) -> t.List[t.Dict]:
        return self.dict()["__root__"]


_faithfulness_output_instructions = get_json_format_instructions(
    StatementFaithfulnessAnswers
)
_faithfulness_output_parser = RagasoutputParser(
    pydantic_object=StatementFaithfulnessAnswers
)

NLI_STATEMENTS_MESSAGE = Prompt(
    name="nli_statements",
    instruction="Your task is to judge the faithfulness of a series of statements based on a given context. For each statement you must return verdict as 1 if the statement can be directly inferred based on the context or 0 if the statement can not be directly inferred based on the context.",
    output_format_instruction=_faithfulness_output_instructions,
    examples=[
        {
            "context": """John is a student at XYZ University. He is pursuing a degree in Computer Science. He is enrolled in several courses this semester, including Data Structures, Algorithms, and Database Management. John is a diligent student and spends a significant amount of time studying and completing assignments. He often stays late in the library to work on his projects.""",
            "statements": [
                "John is majoring in Biology.",
                "John is taking a course on Artificial Intelligence.",
                "John is a dedicated student.",
                "John has a part-time job.",
            ],
            "answer": StatementFaithfulnessAnswers.parse_obj(
                [
                    {
                        "statement": "John is majoring in Biology.",
                        "reason": "John's major is explicitly mentioned as Computer Science. There is no information suggesting he is majoring in Biology.",
                        "verdict": 0,
                    },
                    {
                        "statement": "John is taking a course on Artificial Intelligence.",
                        "reason": "The context mentions the courses John is currently enrolled in, and Artificial Intelligence is not mentioned. Therefore, it cannot be deduced that John is taking a course on AI.",
                        "verdict": 0,
                    },
                    {
                        "statement": "John is a dedicated student.",
                        "reason": "The context states that he spends a significant amount of time studying and completing assignments. Additionally, it mentions that he often stays late in the library to work on his projects, which implies dedication.",
                        "verdict": 1,
                    },
                    {
                        "statement": "John has a part-time job.",
                        "reason": "There is no information given in the context about John having a part-time job.",
                        "verdict": 0,
                    },
                ]
            ).dicts(),
        },
        {
            "context": """Photosynthesis is a process used by plants, algae, and certain bacteria to convert light energy into chemical energy.""",
            "statements": ["Albert Einstein was a genius."],
            "answer": StatementFaithfulnessAnswers.parse_obj(
                [
                    {
                        "statement": "Albert Einstein was a genius.",
                        "reason": "The context and statement are unrelated",
                        "verdict": 0,
                    }
                ]
            ).dicts(),
        },
    ],
    input_keys=["context", "statements"],
    output_key="answer",
    output_type="json",
    language="english",
)  # noqa: E501

NEW_NLI_STATEMENTS_MESSAGE = Prompt(
    name="new_nli_statements",
    instruction="您的任务是根据给定的上下文判断一系列语句的忠实性。对于每条语句，如果可以根据上下文直接推断出该语句，则必须返回 1；如果不能根据上下文直接推断出该语句，则必须返回 0。",
    output_format_instruction=_faithfulness_output_instructions,
    examples=[
        {
            "context": """约翰是 XYZ 大学的学生。他正在攻读计算机科学学位。本学期他选修了几门课程，包括数据结构、算法和数据库管理。约翰是个勤奋的学生，花了大量时间学习和完成作业。他经常在图书馆待到很晚，以完成他的项目。""",
            "statements": [
                "约翰主修生物学.",
                "约翰正在学习人工智能课程.",
                "约翰是一个敬业的学生。",
                "约翰有一份兼职工作。",
            ],
            "answer": StatementFaithfulnessAnswers.parse_obj(
                [
                    {
                        "statement": "约翰主修生物学.",
                        "reason": "约翰的主修专业明确为计算机科学。没有任何信息表明他主修生物学。",
                        "verdict": 0,
                    },
                    {
                        "statement": "约翰正在学习人工智能课程..",
                        "reason": "上下文提到了约翰目前正在学习的课程，但没有提到人工智能。因此，不能推断约翰正在学习人工智能课程。",
                        "verdict": 0,
                    },
                    {
                        "statement": "约翰是一个敬业的学生.",
                        "reason": "上下文说他花了大量时间学习和完成作业。此外，上下文还提到他经常在图书馆工作到很晚，这意味着他很投入。",
                        "verdict": 1,
                    },
                    {
                        "statement": "约翰有一份兼职工作",
                        "reason": "上下文中没有关于约翰有兼职工作的信息.",
                        "verdict": 0,
                    },
                ]
            ).dicts(),
        },
        {
            "context": """光合作用是植物、藻类和某些细菌将光能转化为化学能的过程。""",
            "statements": ["阿尔伯特-爱因斯坦是个天才。"],
            "answer": StatementFaithfulnessAnswers.parse_obj(
                [
                    {
                        "statement": "阿尔伯特-爱因斯坦是个天才。.",
                        "reason": "上下文与陈述无关",
                        "verdict": 0,
                    }
                ]
            ).dicts(),
        },
    ],
    input_keys=["context", "statements"],
    output_key="answer",
    output_type="json",
    language="chinese",
)  # noqa: E501


@dataclass
class Faithfulness(MetricWithLLM):
    name: str = "faithfulness"  # type: ignore
    evaluation_mode: EvaluationMode = EvaluationMode.qac  # type: ignore
    nli_statements_message: Prompt = field(
        default_factory=lambda: NEW_NLI_STATEMENTS_MESSAGE
    )
    statement_prompt: Prompt = field(default_factory=lambda: LONG_FORM_ANSWER_STATEMENTS_PROMPT)
    sentence_segmenter: t.Optional[HasSegmentMethod] = None
    max_retries: int = 1
    _reproducibility: int = 1

    @property
    def reproducibility(self):
        return self._reproducibility

    @reproducibility.setter
    def reproducibility(self, value):
        if value < 1:
            logger.warning("reproducibility cannot be less than 1, setting to 1")
            value = 1
        elif value % 2 == 0:
            logger.warning(
                "reproducibility level cannot be set to even number, setting to odd"
            )
            value += 1
        self._reproducibility = value

    def __post_init__(self):
        if self.sentence_segmenter is None:
            language = self.nli_statements_message.language
            self.sentence_segmenter = get_segmenter(language=language, clean=False)

    def _create_nli_prompt(self, row: t.Dict, statements: t.List[str]) -> PromptValue:
        assert self.llm is not None, "llm must be set to compute score"

        contexts = row["contexts"]
        # check if the statements are support in the contexts
        contexts_str: str = "\n".join(contexts)
        statements_str: str = json.dumps(statements, ensure_ascii=False)
        prompt_value = self.nli_statements_message.format(
            context=contexts_str, statements=statements_str
        )
        return prompt_value

    def _create_statements_prompt(self, row: t.Dict) -> PromptValue:
        assert self.sentence_segmenter is not None, "sentence_segmenter is not set"

        text, question = row["answer"], row["question"]
        # sentences = self.sentence_segmenter.segment(text)
        # sentences = [
        #     sentence for sentence in sentences if sentence.strip().endswith(".")
        # ]
        # sentences = "\n".join([f"{i}:{x}" for i, x in enumerate(sentences)])
        sentences = row["answer"]
        prompt_value = self.statement_prompt.format(
            question=question, answer=text, sentences=sentences
        )
        return prompt_value

    def _compute_score(self, answers: StatementFaithfulnessAnswers):
        # check the verdicts and compute the score
        faithful_statements = sum(
            1 if answer.verdict else 0 for answer in answers.__root__
        )
        num_statements = len(answers.__root__)
        if num_statements:
            score = faithful_statements / num_statements
        else:
            logger.warning("No statements were generated from the answer.")
            score = np.nan

        return score

    async def _ascore(self: t.Self, row: t.Dict, callbacks: Callbacks) -> t.Dict:
        """
        returns the NLI score for each (q, c, a) pair
        """
        assert self.llm is not None, "LLM is not set"

        p_value = self._create_statements_prompt(row)
        statements = await self.llm.generate(
            p_value,
            callbacks=callbacks,
        )
        statements = await _statements_output_parser.aparse(
            statements.generations[0][0].text, p_value, self.llm, self.max_retries
        )

        if statements is None:
            return {"faithfulness_list": [], "average_verdict": np.nan}

        statements = [item["simpler_statements"] for item in statements.dicts()]
        statements = [item for sublist in statements for item in sublist]

        assert isinstance(statements, t.List), "statements must be a list"

        p_value = self._create_nli_prompt(row, statements)
        nli_result = await self.llm.generate(
            p_value,
            callbacks=callbacks,
            n=self._reproducibility,
        )

        nli_result_text = [
            nli_result.generations[0][i].text for i in range(self._reproducibility)
        ]
        faithfulness_list = [
            await _faithfulness_output_parser.aparse(
                text, p_value, self.llm, self.max_retries
            )
            for text in nli_result_text
        ]

        faithfulness_list = [
            faith.dicts() for faith in faithfulness_list if faith is not None
        ]

        # if faithfulness_list:
        #     faithfulness_list = ensembler.from_discrete(
        #         faithfulness_list,
        #         "verdict",
        #     )
        #
        #     faithfulness_list = StatementFaithfulnessAnswers.parse_obj(
        #         faithfulness_list
        #     )
        # else:
        #     return np.nan

        verdicts = [item['verdict'] for sublist in faithfulness_list for item in sublist]

        # 计算平均数
        if verdicts:
            average_verdict = sum(verdicts) / len(verdicts)
        else:
            average_verdict = 0

        # print(f"result: {result}, type: {type(result)}")
        # print(f"faithfulness_list:{ faithfulness_list}")
        return {"faithfulness_list": faithfulness_list, "scores": float(average_verdict)}

    def adapt(self, language: str, cache_dir: t.Optional[str] = None) -> None:
        assert self.llm is not None, "LLM is not set"

        logger.info(f"Adapting Faithfulness metric to {language}")

        self.nli_statements_message = self.nli_statements_message.adapt(
            language, self.llm, cache_dir
        )
        self.statement_prompt = self.statement_prompt.adapt(
            language, self.llm, cache_dir
        )

        # self.sentence_segmenter = get_segmenter(language=language, clean=False)

    def save(self, cache_dir: t.Optional[str] = None) -> None:
        self.nli_statements_message.save(cache_dir)
        self.statement_prompt.save(cache_dir)


@dataclass
class FaithulnesswithHHEM(Faithfulness):
    name: str = "faithfulness_with_hhem"  # type: ignore

    def __post_init__(self):
        try:
            from transformers import AutoModelForSequenceClassification
        except ImportError:
            raise ImportError(
                "Huggingface transformers must be installed to use this feature, try `pip install transformers`"
            )
        self.nli_classifier = AutoModelForSequenceClassification.from_pretrained(
            "vectara/hallucination_evaluation_model", trust_remote_code=True
        )
        super().__post_init__()

    def _create_pairs(
        self, row: t.Dict, statements: t.List[str]
    ) -> t.List[t.Tuple[str, str]]:
        """
        create pairs of (question, answer) from the row
        """
        premise = "\n".join(row["contexts"])
        pairs = [(premise, statement) for statement in statements]
        return pairs

    async def _ascore(self: t.Self, row: t.Dict, callbacks: Callbacks) -> float:
        """
        returns the NLI score for each (q, c, a) pair
        """
        assert self.llm is not None, "LLM is not set"

        p_value = self._create_statements_prompt(row)
        statements = await self.llm.generate(
            p_value,
            callbacks=callbacks,
        )
        statements = await _statements_output_parser.aparse(
            statements.generations[0][0].text, p_value, self.llm, self.max_retries
        )

        if statements is None:
            return np.nan

        statements = [item["simpler_statements"] for item in statements.dicts()]
        statements = [item for sublist in statements for item in sublist]

        assert isinstance(statements, t.List), "statements must be a list"

        pairs = self._create_pairs(row, statements)
        scores = self.nli_classifier.predict(pairs).detach().numpy().round()
        return scores.sum() / len(scores)


faithfulness = Faithfulness()

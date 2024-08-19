from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass, field

import numpy as np
from langchain.pydantic_v1 import BaseModel, Field

from ragas.llms.output_parser import RagasoutputParser, get_json_format_instructions
from ragas.llms.prompt import Prompt, PromptValue
from ragas.metrics.base import EvaluationMode, MetricWithLLM, ensembler

if t.TYPE_CHECKING:
    from langchain_core.callbacks import Callbacks

logger = logging.getLogger(__name__)


class ContextPrecisionVerification(BaseModel):
    """Answer for the verification task wether the context was useful."""

    reason: str = Field(..., description="Reason for verification")
    verdict: int = Field(..., description="Binary (0/1) verdict of verification")


class ContextPrecisionVerifications(BaseModel):
    __root__: t.List[ContextPrecisionVerification]


_verification_output_instructions = get_json_format_instructions(
    ContextPrecisionVerification
)
_output_parser = RagasoutputParser(pydantic_object=ContextPrecisionVerification)

CONTEXT_PRECISION = Prompt(
    name="context_precision",
    instruction="""Given question, answer and context verify if the context was useful in arriving at the given answer. Give verdict as "1" if useful and "0" if not with json output.""",
    output_format_instruction=_verification_output_instructions,
    examples=[
        {
            "question": """What can you tell me about albert Albert Einstein?""",
            "context": """Albert Einstein (14 March 1879 – 18 April 1955) was a German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time. Best known for developing the theory of relativity, he also made important contributions to quantum mechanics, and was thus a central figure in the revolutionary reshaping of the scientific understanding of nature that modern physics accomplished in the first decades of the twentieth century. His mass–energy equivalence formula E = mc2, which arises from relativity theory, has been called "the world's most famous equation". He received the 1921 Nobel Prize in Physics "for his services to theoretical physics, and especially for his discovery of the law of the photoelectric effect", a pivotal step in the development of quantum theory. His work is also known for its influence on the philosophy of science. In a 1999 poll of 130 leading physicists worldwide by the British journal Physics World, Einstein was ranked the greatest physicist of all time. His intellectual achievements and originality have made Einstein synonymous with genius.""",
            "answer": """Albert Einstein born in 14 March 1879 was German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time. He received the 1921 Nobel Prize in Physics for his services to theoretical physics. He published 4 papers in 1905. Einstein moved to Switzerland in 1895""",
            "verification": ContextPrecisionVerification(
                reason="The provided context was indeed useful in arriving at the given answer. The context includes key information about Albert Einstein's life and contributions, which are reflected in the answer.",
                verdict=1,
            ).dict(),
        },
        {
            "question": """who won 2020 icc world cup?""",
            "context": """The 2022 ICC Men's T20 World Cup, held from October 16 to November 13, 2022, in Australia, was the eighth edition of the tournament. Originally scheduled for 2020, it was postponed due to the COVID-19 pandemic. England emerged victorious, defeating Pakistan by five wickets in the final to clinch their second ICC Men's T20 World Cup title.""",
            "answer": """England""",
            "verification": ContextPrecisionVerification(
                reason="the context was useful in clarifying the situation regarding the 2020 ICC World Cup and indicating that England was the winner of the tournament that was intended to be held in 2020 but actually took place in 2022.",
                verdict=1,
            ).dict(),
        },
        {
            "question": """What is the tallest mountain in the world?""",
            "context": """The Andes is the longest continental mountain range in the world, located in South America. It stretches across seven countries and features many of the highest peaks in the Western Hemisphere. The range is known for its diverse ecosystems, including the high-altitude Andean Plateau and the Amazon rainforest.""",
            "answer": """Mount Everest.""",
            "verification": ContextPrecisionVerification(
                reason="the provided context discusses the Andes mountain range, which, while impressive, does not include Mount Everest or directly relate to the question about the world's tallest mountain.",
                verdict=0,
            ).dict(),
        },
    ],
    input_keys=["question", "context", "answer"],
    output_key="verification",
    output_type="json",
)


CONTEXT_PRECISION_NEW = Prompt(
    name="context_precision_new",
    instruction="""给定问题、答案和上下文，验证上下文在得出给定答案时是否有用。如果有用，给出判决为 "1"，如果无用，则为 "0"，并以 JSON 格式输出。""",
    output_format_instruction=_verification_output_instructions,
    examples=[
        {
            "question": """你能告诉我关于阿尔伯特·爱因斯坦的事情吗？""",
            "context": """阿尔伯特·爱因斯坦（1879年3月14日 - 1955年4月18日）是一位出生于德国的理论物理学家，被广泛认为是有史以来最伟大和最有影响力的科学家之一。他最著名的是发展了相对论理论，同时也对量子力学做出了重要贡献，因此在二十世纪前几十年现代物理学重新塑造科学理解自然的过程中，他是一个核心人物。他的质能等效公式E=mc²，源自相对论理论，被称为“世界上最著名的方程”。他因“在理论物理学方面的贡献，特别是发现了光电效应法则”而获得了1921年的诺贝尔物理学奖，这是量子理论发展的一个关键步骤。他的工作也因其对科学哲学的影响而闻名。在1999年由英国杂志《物理世界》对全球130位领先物理学家进行的调查中，爱因斯坦被评为有史以来最伟大的物理学家。他的智力成就和原创性使爱因斯坦成为天才的代名词。""",
            "answer": """阿尔伯特·爱因斯坦出生于1879年3月14日，是一位出生于德国的理论物理学家，被广泛认为是有史以来最伟大和最有影响力的科学家之一。他因在理论物理学方面的贡献而获得了1921年的诺贝尔物理学奖。他在1905年发表了4篇论文。爱因斯坦于1895年搬到瑞士。""",
            "verification": ContextPrecisionVerification(
                reason="提供的上下文在得出给定答案时确实有用。上下文包括关于阿尔伯特·爱因斯坦的生活和贡献的关键信息，这些信息在答案中得到了反映。",
                verdict=1,
            ).dict(),
        },
        {
            "question": """谁赢得了2020年国际板球理事会世界杯？""",
            "context": """2022年国际板球理事会男子T20世界杯于2022年10月16日至11月13日在澳大利亚举行，这是该赛事的第八届。原定于2020年举行，但由于COVID-19大流行而推迟。英格兰在决赛中击败巴基斯坦，以五个小门的优势胜出，赢得了他们的第二个国际板球理事会男子T20世界杯冠军。""",
            "answer": """英格兰""",
            "verification": ContextPrecisionVerification(
                reason="上下文有助于澄清2020年国际板球理事会世界杯的情况，并指出英格兰是原定于2020年举行但实际上在2022年举行的比赛的冠军。",
                verdict=1,
            ).dict(),
        },
        {
            "question": """世界上最高的山是什么？""",
            "context": """安第斯山脉是世界上最长的大陆山脉，位于南美洲。它横跨七个国家，拥有西半球许多最高的山峰。该山脉以其多样的生态系统而闻名，包括高海拔的安第斯高原和亚马逊雨林。""",
            "answer": """珠穆朗玛峰。""",
            "verification": ContextPrecisionVerification(
                reason="提供的上下文讨论了安第斯山脉，尽管令人印象深刻，但不包括珠穆朗玛峰，也不直接与关于世界上最高山峰的问题相关。",
                verdict=0,
            ).dict(),
        },
    ],
    input_keys=["question", "context", "answer"],
    output_key="verification",
    output_type="json",
)

@dataclass
class ContextPrecision(MetricWithLLM):
    """
    Average Precision is a metric that evaluates whether all of the
    relevant items selected by the model are ranked higher or not.

    Attributes
    ----------
    name : str
    evaluation_mode: EvaluationMode
    context_precision_prompt: Prompt
    """

    name: str = "context_precision"  # type: ignore
    evaluation_mode: EvaluationMode = EvaluationMode.qcg  # type: ignore
    context_precision_prompt: Prompt = field(default_factory=lambda: CONTEXT_PRECISION_NEW)
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

    def _get_row_attributes(self, row: t.Dict) -> t.Tuple[str, t.List[str], t.Any]:
        return row["question"], row["contexts"], row["ground_truth"]

    def _context_precision_prompt(self, row: t.Dict) -> t.List[PromptValue]:
        question, contexts, answer = self._get_row_attributes(row)
        return [
            self.context_precision_prompt.format(
                question=question, context=c, answer=answer
            )
            for c in contexts
        ]

    def _calculate_average_precision(
        self, verifications: t.List[ContextPrecisionVerification]
    ) -> float:
        score = np.nan

        verdict_list = [1 if ver.verdict else 0 for ver in verifications]
        denominator = sum(verdict_list) + 1e-10
        numerator = sum(
            [
                (sum(verdict_list[: i + 1]) / (i + 1)) * verdict_list[i]
                for i in range(len(verdict_list))
            ]
        )
        score = numerator / denominator
        if np.isnan(score):
            logger.warning(
                "Invalid response format. Expected a list of dictionaries with keys 'verdict'"
            )
        return score

    async def _ascore(
        self: t.Self,
        row: t.Dict,
        callbacks: Callbacks,
    ) -> t.Dict:
        assert self.llm is not None, "LLM is not set"

        human_prompts = self._context_precision_prompt(row)
        responses = []
        for hp in human_prompts:
            results = await self.llm.generate(
                hp,
                callbacks=callbacks,
                n=self.reproducibility,
            )
            results = [
                await _output_parser.aparse(item.text, hp, self.llm, self.max_retries)
                for item in results.generations[0]
            ]

            responses.append(
                [result.dict() for result in results if result is not None]
            )

        answers = []
        for response in responses:
            agg_answer = ensembler.from_discrete([response], "verdict")
            if agg_answer:
                agg_answer = ContextPrecisionVerification.parse_obj(agg_answer[0])
            answers.append(agg_answer)
        answer = [answer.dict() for answer in answers]

        answers = ContextPrecisionVerifications(__root__=answers)
        score = self._calculate_average_precision(answers.__root__)


        return {"context_precision_list": answer, "scores": score}

    def adapt(self, language: str, cache_dir: str | None = None) -> None:
        assert self.llm is not None, "LLM is not set"

        logging.info(f"Adapting Context Precision to {language}")
        self.context_precision_prompt = self.context_precision_prompt.adapt(
            language, self.llm, cache_dir
        )

    def save(self, cache_dir: str | None = None) -> None:
        self.context_precision_prompt.save(cache_dir)


@dataclass
class ContextUtilization(ContextPrecision):
    name: str = "context_utilization"
    evaluation_mode: EvaluationMode = EvaluationMode.qac

    def _get_row_attributes(self, row: t.Dict) -> t.Tuple[str, t.List[str], t.Any]:
        return row["question"], row["contexts"], row["answer"]


context_precision = ContextPrecision()
context_utilization = ContextUtilization()

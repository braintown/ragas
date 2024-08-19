from __future__ import annotations

import logging
import typing as t
from dataclasses import dataclass, field
from typing import Dict
import ast

import numpy as np
from langchain.pydantic_v1 import BaseModel

from ragas.llms.output_parser import RagasoutputParser, get_json_format_instructions
from ragas.llms.prompt import Prompt
from ragas.metrics.base import EvaluationMode, MetricWithLLM
import requests

if t.TYPE_CHECKING:
    from langchain.callbacks.base import Callbacks

logger = logging.getLogger(__name__)


class ContextEntitiesResponse(BaseModel):
    entities: t.List[str]


_output_instructions = get_json_format_instructions(
    pydantic_object=ContextEntitiesResponse
)
_output_parser = RagasoutputParser(pydantic_object=ContextEntitiesResponse)

TEXT_ENTITY_EXTRACTION = Prompt(
    name="text_entity_extraction",
    instruction="""Given a text, extract unique entities without repetition. Ensure you consider different forms or mentions of the same entity as a single entity.""",
    input_keys=["text"],
    output_key="output",
    output_type="json",
    output_format_instruction=_output_instructions,
    examples=[
        {
            "text": """The Eiffel Tower, located in Paris, France, is one of the most iconic landmarks globally.
            Millions of visitors are attracted to it each year for its breathtaking views of the city.
            Completed in 1889, it was constructed in time for the 1889 World's Fair.""",
            "output": ContextEntitiesResponse.parse_obj(
                {
                    "entities": [
                        "Eiffel Tower",
                        "Paris",
                        "France",
                        "1889",
                        "World's Fair",
                    ],
                }
            ).dict(),
        },
        {
            "text": """The Colosseum in Rome, also known as the Flavian Amphitheatre, stands as a monument to Roman architectural and engineering achievement.
            Construction began under Emperor Vespasian in AD 70 and was completed by his son Titus in AD 80.
            It could hold between 50,000 and 80,000 spectators who watched gladiatorial contests and public spectacles.""",
            "output": ContextEntitiesResponse.parse_obj(
                {
                    "entities": [
                        "Colosseum",
                        "Rome",
                        "Flavian Amphitheatre",
                        "Vespasian",
                        "AD 70",
                        "Titus",
                        "AD 80",
                    ],
                }
            ).dict(),
        },
        {
            "text": """The Great Wall of China, stretching over 21,196 kilometers from east to west, is a marvel of ancient defensive architecture.
            Built to protect against invasions from the north, its construction started as early as the 7th century BC.
            Today, it is a UNESCO World Heritage Site and a major tourist attraction.""",
            "output": ContextEntitiesResponse.parse_obj(
                {
                    "entities": [
                        "Great Wall of China",
                        "21,196 kilometers",
                        "7th century BC",
                        "UNESCO World Heritage Site",
                    ],
                }
            ).dict(),
        },
        {
            "text": """The Apollo 11 mission, which launched on July 16, 1969, marked the first time humans landed on the Moon.
            Astronauts Neil Armstrong, Buzz Aldrin, and Michael Collins made history, with Armstrong being the first man to step on the lunar surface.
            This event was a significant milestone in space exploration.""",
            "output": ContextEntitiesResponse.parse_obj(
                {
                    "entities": [
                        "Apollo 11 mission",
                        "July 16, 1969",
                        "Moon",
                        "Neil Armstrong",
                        "Buzz Aldrin",
                        "Michael Collins",
                    ],
                }
            ).dict(),
        },
    ],
)


TEXT_ENTITY_EXTRACTION_NEW = Prompt(
    name="text_entity_extraction_new",
    instruction="""给定一段文本，提取唯一的实体，不重复。确保将同一实体的不同形式或提及视为单一实体。""",
    input_keys=["text"],
    output_key="output",
    output_type="json",
    output_format_instruction=_output_instructions,
    examples=[
        {
            "text": """埃菲尔铁塔，位于法国巴黎，是全球最具标志性的地标之一。
            每年都有数百万游客被它的城市美景所吸引。
            它于1889年完工，正好赶上1889年的世界博览会。""",
            "output": ContextEntitiesResponse.parse_obj(
                {
                    "entities": [
                        "埃菲尔铁塔",
                        "巴黎",
                        "法国",
                        "1889",
                        "世界博览会",
                    ],
                }
            ).dict(),
        },
        {
            "text": """罗马的斗兽场，也被称为弗拉维安圆形剧场，作为罗马建筑和工程成就的纪念碑而矗立。
            建设始于公元70年由皇帝韦斯帕西安开始，由他的儿子提图斯于公元80年完成。
            它可以容纳5万到8万名观众观看角斗士比赛和公众表演。""",
            "output": ContextEntitiesResponse.parse_obj(
                {
                    "entities": [
                        "斗兽场",
                        "罗马",
                        "弗拉维安圆形剧场",
                        "韦斯帕西安",
                        "公元70年",
                        "提图斯",
                        "公元80年",
                    ],
                }
            ).dict(),
        },
        {
            "text": """中国的长城，从东到西绵延21,196公里，是古代防御建筑的奇迹。
            为了防止来自北方的入侵，其建设早在公元前7世纪就开始了。
            今天，它是联合国教科文组织世界遗产和主要旅游景点。""",
            "output": ContextEntitiesResponse.parse_obj(
                {
                    "entities": [
                        "长城",
                        "21,196公里",
                        "公元前7世纪",
                        "联合国教科文组织世界遗产",
                    ],
                }
            ).dict(),
        },
        {
            "text": """阿波罗11号任务于1969年7月16日发射，标志着人类首次登月。
            宇航员尼尔·阿姆斯特朗、巴兹·奥尔德林和迈克尔·柯林斯创造了历史，其中阿姆斯特朗成为第一个踏上月球表面的人。
            这一事件是太空探索的一个重要里程碑。""",
            "output": ContextEntitiesResponse.parse_obj(
                {
                    "entities": [
                        "阿波罗11号任务",
                        "1969年7月16日",
                        "月球",
                        "尼尔·阿姆斯特朗",
                        "巴兹·奥尔德林",
                        "迈克尔·柯林斯",
                    ],
                }
            ).dict(),
        },
    ],
)

@dataclass
class ContextEntityRecall(MetricWithLLM):
    """
    Calculates recall based on entities present in ground truth and context.
    Let CN be the set of entities present in context,
    GN be the set of entities present in the ground truth.

    Then we define can the context entity recall as follows:
    Context Entity recall = | CN ∩ GN | / | GN |

    If this quantity is 1, we can say that the retrieval mechanism has
    retrieved context which covers all entities present in the ground truth,
    thus being a useful retrieval. Thus this can be used to evaluate retrieval
    mechanisms in specific use cases where entities matter, for example, a
    tourism help chatbot.

    Attributes
    ----------
    name : str
    batch_size : int
        Batch size for openai completion.
    """

    name: str = "context_entity_recall"  # type: ignore
    evaluation_mode: EvaluationMode = EvaluationMode.gc  # type: ignore
    context_entity_recall_prompt: Prompt = field(
        default_factory=lambda: TEXT_ENTITY_EXTRACTION_NEW
    )
    batch_size: int = 15
    max_retries: int = 1

    @staticmethod
    def is_similar(entity1: t.Sequence[str], entity2: t.Sequence[str]) -> str:
        group_id = "1693463149758508"
        api_key = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJHcm91cE5hbWUiOiLkuIrmtbfpm7fluqbnvZHnu5znp5HmioDmnInpmZDlhazlj7giLCJVc2VyTmFtZSI6IuaigeaeqyIsIkFjY291bnQiOiIiLCJTdWJqZWN0SUQiOiIxNjkzNDYzMTQ5ODMwNjAyIiwiUGhvbmUiOiIxNzY3NzMxMzE5NyIsIkdyb3VwSUQiOiIxNjkzNDYzMTQ5NzU4NTA4IiwiUGFnZU5hbWUiOiIiLCJNYWlsIjoiQUlSREBnZW1wb2xsLmNvbSIsIkNyZWF0ZVRpbWUiOiIyMDI0LTAxLTI0IDE5OjAzOjM4IiwiaXNzIjoibWluaW1heCJ9.tKy7T6gagvMp545jm2tOG42LbLm3SYFAQ4MlkXAMBlzYoeyy94VWls9ZFcVkPX9WFigvqsHyvlk6UIBpasOoQ3eKu5YopCYrO6j3eqNZdbUNVfUe1yDwNFLZN04Eu_sRfaa5uqYbqgFsl8k9kNleX4y9rh7fni-Hkx1e0LOnXWVVUbMcdChvpBuq_QhruodhFysxKvrWgV9PVqkaLS7Ym6N2J1w1TrhcH9yKZ6wyZi0GFvUknY0pld_E7B-Hn51G0ClM2fHGsMb7s5rb0NYWA_v4S1GsWB5-91mVzdfu3XPUPo9MJ4O_wZbf398HKsjd9cg52R1e-ho321RndNIvXw"
        url = "https://api.minimax.chat/v1/text/chatcompletion_pro?GroupId=" + group_id
        headers = {"Content-Type": "application/json", "Authorization": "Bearer " + api_key}

        payload = {
            "bot_setting": [
                {
                    "bot_name": "关键词",
                    "content": "关键词帮助分析词组意思,能够理解各种词组之间的关联性",
                }
            ],
            "messages": [{"sender_type": "USER", "sender_name": "小明",
                          "text": f"判断以下两个列表是否具有相似的词组意思:\n1. {entity1}\n2. {entity2}\n.有的话提取出具体的词组列表,注意只返回第一个列表中的内容，不需要分析过程.例子如下："
                                  f"ground_truth的列表为 ['凯迪拉克CT5', '全景天窗', '126色氛围灯'], contexts的列表为: ['全新CT5', '33英寸9K超视网膜曲面屏','Mini-LED', '126色设计师甄选氛围灯', '16扬声器AKG录音棚级音响'],"
                                  f"返回['凯迪拉克CT5', '126色氛围灯'], 不要返回['凯迪拉克CT5', '126色设计师甄选氛围灯']"
                                  f"知识补充：凯迪拉克傲歌和IQ傲歌为同一辆车型，凯迪拉克CT5和全新CT5为一辆车型"
                                  f"当列表没有相似的词组意思，返回空列表[]"}],
            "reply_constraints": {"sender_type": "BOT", "sender_name": "关键词"},
            "model": "abab6.5s-chat",
            "tokens_to_generate": 2048,
            "temperature": 0.01,
            "top_p": 0.95,
        }

        response = requests.request("POST", url, headers=headers, json=payload)

        return response.json()['reply']

    def _compute_score(
            self, ground_truth_entities: t.Sequence[str], context_entities: t.Sequence[str]
    ) -> (float, list):
        similar_list = self.is_similar(ground_truth_entities, context_entities)
        if type(similar_list) == str:
            index= similar_list.find('[')
            if index != -1:
                try:
                    similar_list = ast.literal_eval(similar_list)
                except Exception as e:
                    print(e)
                    similar_list = []
            else:
                similar_list = []
        else:
            similar_list = []
        score = len(similar_list) / (len(ground_truth_entities) + 1e-8)
        return score, similar_list

    async def get_entities(
            self,
            text: str,
            callbacks: Callbacks,
    ) -> t.Optional[ContextEntitiesResponse]:
        assert self.llm is not None, "LLM is not initialized"
        p_value = self.context_entity_recall_prompt.format(
            text=text,
        )
        result = await self.llm.generate(
            prompt=p_value,
            callbacks=callbacks,
        )

        result_text = result.generations[0][0].text
        answer = await _output_parser.aparse(
            result_text, p_value, self.llm, self.max_retries
        )
        if answer is None:
            return ContextEntitiesResponse(entities=[])

        return answer

    async def _ascore(
            self,
            row: Dict,
            callbacks: Callbacks,
    ) -> t.Dict:
        ground_truth, contexts = row["ground_truth"], row["contexts"]
        ground_truth = await self.get_entities(ground_truth, callbacks=callbacks)
        contexts = await self.get_entities("\n".join(contexts), callbacks=callbacks)
        if ground_truth is None or contexts is None:
            return {"ground_truth is None or contexts is None": None}
        score, intersecting_entities = self._compute_score(ground_truth.entities, contexts.entities)
        context_entity_recall_list = [
            f"same_sequence:{intersecting_entities},contexts_entities:{contexts.entities},ground_truth_entities:{ground_truth.entities}"]

        return {"context_entity_recall_list": context_entity_recall_list, "scores": float(score)}

    def save(self, cache_dir: str | None = None) -> None:
        return self.context_entity_recall_prompt.save(cache_dir)


context_entity_recall = ContextEntityRecall(batch_size=15)

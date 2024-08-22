from datasets import Dataset
from ragas import evaluate
import os
import json

from langchain_openai import AzureChatOpenAI
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from langchain_openai import ChatOpenAI
from ragas.metrics import faithfulness, answer_relevancy, answer_correctness, context_entity_recall, context_precision, \
    context_recall,answer_similarity
from ragas import adapt

os.environ["AZURE_OPENAI_API_KEY"] = "aa34c68036574cef8c9490d3fe9d4cd3"
azure_configs = {
    "base_url": "https://aigc-dev-gempoll.openai.azure.com/",
    "model_deployment": "gpt4o",
    "model_name": "gpt4o",
    "embedding_deployment": "embedding-large",
    "embedding_name": "embedding-large",  # most likely
}
azure_model = AzureChatOpenAI(
    openai_api_version="2024-06-01",
    azure_endpoint=azure_configs["base_url"],
    azure_deployment=azure_configs["model_deployment"],
    model=azure_configs["model_name"],
    validate_base_url=False,
)
azure_embeddings = AzureOpenAIEmbeddings(
    openai_api_version="2024-06-01",
    azure_endpoint=azure_configs["base_url"],
    azure_deployment=azure_configs["embedding_deployment"],
    model=azure_configs["embedding_name"],
)


data_samples = {
    "question": [
        "全新CT5在长途驾驶时的座椅舒适性如何？",
        "傲歌在长途驾驶中的电池续航表现如何？",
    ],
    "answer": [
        "全新CT5的座椅舒适性非常棒，尤其是在长途驾驶时。它配备了座椅加热、通风和腰部支撑功能，可以有效减轻疲劳。想亲自感受一下吗？",
        "傲歌的续航里程跟驾驶习惯密切相关。电耗表现相当出色，每度电能跑8公里。如果你驾驶习惯良好，续航里程会更加实在，享受长途驾驶也无压力。需具体了解续航情况，也可以来店试驾体验一下哦。"
    ],
    "contexts": [
        [
            "question:为什么CT5的乘坐体验被认为比其他品牌更舒适？ \nanswer:CT5的乘坐体验被认为更舒适是因为它拥有多维度的豪华座舱配置，包括ANC主动降噪、座椅加热通风、腰部支撑等。这些配置提供了比国产车和合资车更高级的乘坐体验，甚至比飞机头等舱还要豪华。",
            "question:全新CT5豪华型的智能座舱有哪些特点和优势？ \nanswer:全新CT5豪华型的智能座舱标配9K33英寸环幕式超视网膜屏，提供了智能化的驾驶体验，提升了车内的科技感和豪华感。",
            "question:全新CT5在性能和驾驶体验方面有哪些特点？ \nanswer:全新CT5在性能和驾驶体验方面的特点包括豪华后驱和超跑级运动套装，配备了mLSD（机械限滑差速器）。"
        ],
        [
            "question:IQ傲歌的续航里程与什么因素深度关联？ \nanswer:IQ傲歌的续航里程与驾驶习惯深度关联，续航里程真实可靠不虚标，实际电耗非常低。",
            "question:IQ傲歌每度电能跑多少公里？ \nanswer:IQ傲歌每度电能跑8公里。",
            "question:IQ傲歌的电耗表现如何？ \nanswer:IQ傲歌的电耗表现非常出色，一度电可以行驶8公里，而行业平均水平大约是6公里。如果使用家用充电，相当于每公里仅需4分钱，这个成本甚至比坐地铁还低。"
        ],
    ],
    "ground_truth": [
        "语音识别系统在行驶中是能够准确识别指令的,无需特别提示词。像凯迪拉克CT5和IQ傲歌的智能语音系统都具备强大的识别能力和反应速度。它们不仅能够迅速识别多条不同功能的指令,还支持免唤醒词的直接命令。\n\nCT5的语音控制系统支持720ms到300ms的响应时间,并且在25秒内可以连续完成10条指令。这意味着即使你在行驶中,系统也能够准确地识别和执行命令。\n\nIQ傲歌的车机语音反应速度也十分迅速,可以直接使用指令而无需唤醒,同时支持主副驾驶双音区识别。这保证了行驶中的任何指令都能被快速识别和执行。\n\n所以,无论是凯迪拉克CT5还是IQ傲歌,在行驶中使用语音控制都是非常准确和方便的,无需特别提示词。这不仅提升了驾驶的安全性,也增加了行车便利性。",
        "关于凯迪拉克CT5的车机系统,您可以放心,它的应用支持和使用体验都是相当给力的。全新的CT5配备了高通骁龙8155芯片、33英寸9K环幕式超视网膜屏和AKG的高端音响系统,这些配置在同级别车型中都是佼佼者。\n\n具体来说,音乐和导航应用上,CT5支持Apple CarPlay的无线连接,可以轻松使用iPhone上的导航和音乐应用,比如Apple Maps、高德地图、QQ音乐、Spotify等等。并且它还有百度智能语音互联交互系统,可以通过语音指令操作车机,使用起来特别方便。\n\n而AKG音响系统更是亮点,在尊贵型和铂金型上有16个扬声器和Quantum Logic环绕声技术,听起来就像移动的录音棚,瞬间提升了车内的音乐享受。\n\n至于导航,这个车机系统不仅支持传统的地图应用,还能够通过OTA进行远程更新,保证地图数据始终最新鲜。 语音助手功能也很贴心,不用动手就能搞定目的地搜索和路线规划。\n\n这样我们来看,CT5的车机系统在便捷性、娱乐性和智能性上都非常出色,能够全面提升您的驾驶体验！是不是特别心动呢？我们可以安排一次详细的试驾,让您亲自体验一下这一系列的高科技配置呢？",
    ]

}

# data_samples = json.load(open("datasets_ground_truth.json"))

dataset = Dataset.from_dict(data_samples)

ragas_results = evaluate(dataset=dataset, metrics=[faithfulness, answer_relevancy, answer_correctness, context_entity_recall, context_precision, context_recall,answer_similarity], llm=azure_model, embeddings=azure_embeddings)
print(ragas_results)

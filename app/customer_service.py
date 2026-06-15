import re
from collections import defaultdict, deque

import httpx

from .config import Settings
from .schemas import CustomerServiceChatResponse, CustomerServiceMessage


_HISTORY: dict[str, deque[dict[str, str]]] = defaultdict(lambda: deque(maxlen=10))


AGENT_CATALOG = """
深度觉醒是一家企业智能体 SaaS 公司，核心智能体包括：
1. 文书生成智能体：起诉状、律师函、合同条款、材料格式校验。
2. 风险审查智能体：合同差异、不利条款、合规、售后和投放风险扫描。
3. 多语客服智能体：7x24 咨询、售后、退换货、物流、多语言问答和线索分流。
4. 商品内容智能体：SKU 标题、卖点、详情页、SEO、跨平台商品内容。
5. 热点选题智能体：热点监测、选题推荐、标题、封面话术和脚本角度。
6. 短视频获客智能体：选题、脚本、分镜、图片和 15 秒片段式视频生成。
行业产品包括律智AI、跨境智营、爆文工厂。
"""


async def generate_customer_service_answer(
    cfg: Settings,
    message: str,
    session_id: str = "website",
) -> CustomerServiceChatResponse:
    text = message.strip()
    intent = _detect_intent(text)
    history = list(_HISTORY[session_id])

    if cfg.text_api_key:
        answer = await _deepseek_answer(cfg, text, history, intent)
    else:
        answer = ""

    if not answer:
        answer = _rule_answer(text, intent)

    messages = _split_messages(answer)
    _HISTORY[session_id].append({"role": "user", "content": text})
    _HISTORY[session_id].append({"role": "assistant", "content": answer})

    return CustomerServiceChatResponse(
        session_id=session_id,
        intent=intent,
        emotion="neutral",
        audience="general",
        answer=answer,
        messages=messages,
    )


async def _deepseek_answer(cfg: Settings, message: str, history: list[dict[str, str]], intent: str) -> str:
    system_prompt = f"""
你是“深度觉醒（上海）人工智能科技有限公司”的官网 AI 客服顾问。
说话要像真人顾问：短句、清楚、克制，不要堆概念，不要编造不存在的承诺。
你的任务是帮访客理解企业智能体、推荐合适智能体、引导预约演示。
如果用户问价格，先解释按人数、智能体类型、对话量、系统接入来评估，引导预约。
如果用户想预约，提醒他可以点击页面“预约演示”，并选择想看的智能体。
如果用户问技术接入，可以说明前端 widget 调后端客服接口，后端可接知识库、CRM、订单和人工客服。

{AGENT_CATALOG}

当前意图：{intent}
请直接输出给用户看的中文回复，控制在 2-4 句话。不要输出 JSON。
"""
    payload = {
        "model": cfg.text_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            *history[-6:],
            {"role": "user", "content": message},
        ],
        "temperature": 0.55,
        "max_tokens": 700,
    }
    headers = {"Authorization": f"Bearer {cfg.text_api_key}", "Content-Type": "application/json"}
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(str(cfg.text_api_url), json=payload, headers=headers)
        if resp.status_code >= 400:
            return ""
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return ""


def _detect_intent(text: str) -> str:
    if any(word in text for word in ["预约", "演示", "联系", "试用"]):
        return "booking"
    if any(word in text for word in ["价格", "报价", "多少钱", "收费"]):
        return "price"
    if any(word in text for word in ["客服", "售后", "退换货", "物流", "多语"]):
        return "customer_service"
    if any(word in text for word in ["短视频", "视频", "获客", "分镜"]):
        return "video_agent"
    if any(word in text for word in ["律师", "法律", "合同", "起诉", "审查"]):
        return "legal"
    if any(word in text for word in ["跨境", "电商", "商品", "SKU"]):
        return "commerce"
    if any(word in text for word in ["接入", "接口", "API", "网站", "部署", "技术"]):
        return "technical"
    if any(word in text for word in ["有哪些", "智能体", "产品", "方案"]):
        return "agent_catalog"
    return "chat"


def _rule_answer(text: str, intent: str) -> str:
    if intent == "booking":
        return "可以预约演示。你点页面上的“预约演示”，先选择想看的智能体，再留下公司、姓名和手机号，我们会按你的场景安排演示。"
    if intent == "price":
        return "价格要看使用人数、智能体类型、对话量，以及是否要接 CRM、订单、知识库等系统。建议先预约演示，我们根据你的业务流程给一版更准的配置。"
    if intent == "customer_service":
        return "多语客服智能体适合官网、跨境电商和售后场景。它可以先接待咨询、回答常见问题、处理退换货和物流问题，再把高价值线索或复杂问题转给人工。"
    if intent == "video_agent":
        return "短视频获客智能体会先生成选题、脚本和分镜，再生成图片和短视频片段。现在更适合做 30 秒起步的片段式获客视频，不是一次性硬吐很长的视频。"
    if intent == "legal":
        return "法律场景建议看律智AI和文书/风险审查智能体。它们适合起诉状、律师函、合同条款初稿、合同差异比对和风险提示，最后仍由律师确认。"
    if intent == "commerce":
        return "跨境和电商场景建议看跨境智营、商品内容智能体和多语客服智能体。它们能覆盖商品内容、SEO、售前咨询、物流售后和多语言接待。"
    if intent == "technical":
        return "技术接入可以先从网页客服 widget 开始：前端把用户问题发到 FastAPI，后端再调用大模型和知识库。后续还能接 CRM、订单、物流或人工客服系统。"
    if intent == "agent_catalog":
        return "我们现在主要有文书生成、风险审查、多语客服、商品内容、热点选题和短视频获客智能体。你告诉我行业和最想自动化的流程，我可以帮你先选一个落地入口。"
    if "你好" in text or "在吗" in text:
        return "你好，我在。你可以问我企业智能体、AI 客服、短视频获客，或者直接说你的行业，我帮你判断适合先看哪个方案。"
    return "我理解。你可以再告诉我你的行业、团队规模，以及最想交给 AI 的重复工作，我会帮你判断适合哪个智能体或行业方案。"


def _split_messages(answer: str) -> list[CustomerServiceMessage]:
    parts = [part.strip() for part in re.split(r"\n+|(?<=。)", answer) if part.strip()]
    if not parts:
        parts = [answer.strip() or "我已收到你的问题，会继续为你整理答案。"]
    parts = parts[:3]
    return [
        CustomerServiceMessage(text=part, delay_ms=0 if index == 0 else 900 + index * 450)
        for index, part in enumerate(parts)
    ]

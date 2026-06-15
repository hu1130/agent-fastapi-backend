import asyncio
import json
import re
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any

import httpx
from fastapi import HTTPException

from .config import Settings
from .schemas import Asset, CopyPlan, LongVideoAgentRequest, LongVideoChapter, LongVideoReferenceSet, LongVideoScene, LongVideoScriptResult, WorkflowRequest


DASHSCOPE_IMAGE_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text2image/image-synthesis"
DASHSCOPE_QWEN_IMAGE_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
DASHSCOPE_VIDEO_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/video-generation/video-synthesis"
DASHSCOPE_TASK_URL = "https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"
JIMENG_URL = "https://maas-openapi.wanjiedata.com/api/jimeng/v1"


async def generate_copywriting(settings: Settings, req: WorkflowRequest) -> list[CopyPlan]:
    if not settings.text_api_key:
        raise HTTPException(status_code=400, detail="TEXT_API_KEY is not configured")

    word_hint = {"短": "100字以内，精炼有力", "中": "200-400字，详略得当"}.get(req.word_limit, "500-800字，深度展开")
    scene_count = max(4, min(24, round(req.video_duration / 6)))
    prompt = f"""你是一个专业的自媒体内容导演 + 运营策划。请按照账号人设，为{req.platform}生成{req.copy_count}套可直接交给图片/视频生成模型的内容方案。

【账号人设】
- 品牌/账号：{req.brand_name or "未指定"}
- 一句话定位：{req.brand_tagline or "未指定"}
- 赛道/领域：{req.niche or "未指定"}
- 人设风格：{req.tone}
- 目标人群：{req.audience}
- 风格参考：{req.reference or "无"}

【本次内容要求】
- 发布平台：{req.platform}
- 内容类型：{req.content_type}
- 主题：{req.topic}
- 字数控制：{word_hint}
- 必须包含的要点：{"、".join(req.features)}
- 额外要求：{req.extra_desc or "无"}

请严格返回 JSON 数组，不要 Markdown，不要解释。数组每一项必须包含：
title, hook, subtitle, caption, description, shortText, hashtags, keywords, coverPrompt, videoPrompt, storyboard。
storyboard 生成约 {scene_count} 个镜头，能连成一个约 {req.video_duration} 秒短视频。每个镜头包含 scene, duration, shot, camera, textOverlay, videoPrompt。
重要：videoPrompt 必须是画面描述，不要要求模型生成可读中文文字。"""

    prompt += f"""

专业长视频生成流程要求：
1. 先拆成 shot list：每个镜头 4-10 秒，必须包含人物、场景、机位、动作、光线、对白/旁白、环境音效。
2. 先生成参考图提示：必须包含 characterReferencePrompt、locationReferencePrompt、productReferencePrompt，用来锁定角色、服装、地点、产品的一致性。
3. storyboard 每个镜头必须额外包含 firstFramePrompt 和 lastFramePrompt；firstFramePrompt 用于图片生成，lastFramePrompt 用于下一段视频承接。
4. 镜头连续性：不是每段独立生成，而是后一镜头承接上一镜头的尾帧动作、人物位置、道具和光线。
5. 画面风格：真实商业纪录片/企业案例片，35mm/50mm 镜头语言，自然光或真实办公室灯光，轻微手持/稳定器，避免 AI 假人感。
6. 后期可执行：每个镜头要说明剪辑点、转场方式、字幕内容和声音元素，方便后期统一调色、补转场、配乐、口型修正。
"""

    prompt += f"""

额外导演要求：
- videoPrompt 要写成足够长的连续剧情，不是零散关键词；要说明人物、场景、动作、情绪、镜头衔接和时间推进。
- 如果总时长超过 15 秒，要把故事拆成可拼接的 15 秒左右片段，每段都要延续同一人物、同一空间逻辑和同一视觉风格。
- 画面必须趋近真人实拍：真实办公室/工厂/门店/会议室，真实人物自然表情，真实光线，轻微手持或稳定器运镜。
- 避免一眼 AI：不要塑料皮肤、过度磨皮、夸张景深、赛博背景、漂浮图标、乱码文字、水印、畸形手指、假人感。
- storyboard 每个镜头的 videoPrompt 都要能单独用于图生视频，并能与前后镜头自然拼接。
"""

    payload = {
        "model": settings.text_model,
        "messages": [
            {"role": "system", "content": "你是专业的自媒体内容导演和分镜策划，始终返回合法 JSON 数组。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
        "max_tokens": 5000,
    }
    headers = {"Authorization": f"Bearer {settings.text_api_key}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(str(settings.text_api_url), json=payload, headers=headers)
    if resp.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"Agent1 failed: {resp.text[:300]}")

    raw = resp.json()["choices"][0]["message"]["content"].strip()
    data = _parse_json_array(raw)
    return [_normalize_plan(item, req, i) for i, item in enumerate(data)]


async def generate_images(settings: Settings, req: WorkflowRequest, plans: list[CopyPlan]) -> list[Asset]:
    provider = req.image_provider or settings.image_provider
    if req.image_count <= 0:
        return []
    prompts = build_image_prompts(plans[0], req)
    if provider == "qwen":
        return await _qwen_images(settings, req, prompts)
    if provider == "jimeng":
        return await _jimeng_images(settings, req, prompts)
    return await _dashscope_images(settings, req, prompts)


async def generate_images_from_prompts(settings: Settings, req: WorkflowRequest, prompts: list[str]) -> list[Asset]:
    provider = req.image_provider or settings.image_provider
    if req.image_count <= 0:
        return []
    clean_prompts = [_human_realism_prompt(prompt) for prompt in prompts if prompt.strip()]
    if not clean_prompts:
        clean_prompts = [_human_realism_prompt(f"{req.brand_name or '品牌'}，{req.topic}，真人实拍广告画面")]
    if provider == "qwen":
        return await _qwen_images(settings, req, clean_prompts)
    if provider == "jimeng":
        return await _jimeng_images(settings, req, clean_prompts)
    return await _dashscope_images(settings, req, clean_prompts)


async def generate_video(settings: Settings, req: WorkflowRequest, plans: list[CopyPlan], images: list[Asset] | None = None) -> list[Asset]:
    if not req.generate_video:
        return []
    provider = req.video_provider or settings.video_provider
    prompt = build_video_prompt(plans[0], req)
    if provider == "qwen":
        return await _qwen_image_to_video(settings, req, prompt, images or [])
    if provider == "jimeng":
        url = await _jimeng_video(settings, prompt)
        return [Asset(url=url, label="即梦视频1")] if url else []
    url = await _dashscope_video(settings, prompt)
    return [Asset(url=url, label="百炼视频1")] if url else []


async def generate_video_from_prompt(settings: Settings, req: WorkflowRequest, video_prompt: str, images: list[Asset] | None = None) -> list[Asset]:
    if not req.generate_video:
        return []
    prompt = _human_realism_prompt(video_prompt)
    provider = req.video_provider or settings.video_provider
    if provider == "qwen":
        return await _qwen_image_to_video(settings, req, prompt, images or [])
    if provider == "jimeng":
        url = await _jimeng_video(settings, prompt)
        return [Asset(url=url, label="即梦视频1")] if url else []
    url = await _dashscope_video(settings, prompt)
    return [Asset(url=url, label="百炼视频1")] if url else []


def _human_realism_prompt(prompt: str) -> str:
    return (
        f"{prompt.strip()}\n"
        "风格要求：真人实拍纪录片/企业案例片质感，真实办公室、真实人物、真实光线，"
        "自然肤色，手部和五官正常，动作克制可信，镜头有生活感和轻微手持感。"
        "镜头语言：按 shot list 执行，每个镜头 4-10 秒；保持角色、服装、地点、产品和道具一致。"
        "连续性：每段开头承接上一段尾帧，动作方向、人物站位、光线色温和情绪节奏要接得上。"
        "声音/后期：可保留自然环境声，旁白/字幕/配乐默认后期添加；画面避免生成可读文字。"
        "后期友好：色彩自然，曝光稳定，方便统一调色、补转场、插帧、去闪烁和修手修脸。"
        "避免 AI 质感、塑料皮肤、过度磨皮、夸张景深、赛博背景、漂浮物、乱码文字、Logo 水印、"
        "不自然肢体、畸形手指、过度饱和、假人感、角色漂移、场景突然切换。"
    )


async def generate_long_video_script(settings: Settings, req: LongVideoAgentRequest) -> LongVideoScriptResult:
    if not settings.text_api_key:
        raise HTTPException(status_code=400, detail="TEXT_API_KEY is not configured")

    scene_count = max(4, min(90, round(req.video_duration / req.scene_seconds)))
    topic_parts = [
        f"公司/品牌：{req.company_name}" if req.company_name else "",
        f"行业：{req.industry}" if req.industry else "",
        f"产品/服务：{req.product_name}" if req.product_name else "",
        f"核心卖点：{req.selling_points}" if req.selling_points else "",
        f"目标客户：{req.target_customer}" if req.target_customer else "",
        f"获客目标：{req.goal}" if req.goal else "",
    ]
    topic = "；".join(part for part in topic_parts if part) or "企业 AI 智能体长视频获客内容"
    prompt = f"""你是企业级 AI 长视频流水线导演，不是一次性生成完整视频的模型。请为一个 {req.video_duration} 秒的长视频生成“短片段生成 + 分镜拼接 + 一致性控制 + 后期剪辑”的生产方案。

重要原则：
- 不要假装一次性生成 3 分钟或 10 分钟完整视频。
- 模型负责生成可用镜头，系统负责镜头连续，剪辑负责让它像片子。
- 每个镜头必须是 4-10 秒短片段，后续通过首尾帧、参考图和后期剪辑拼接成长视频。
- 当前只生成脚本、分镜、参考图提示词、逐镜头视频提示词和后期清单，不生成真实视频。

项目信息：
{topic}
发布平台：{req.platform}
风格：{req.tone}
补充要求：{req.extra_desc or "无"}

流水线要求：
1. 先拆成 shot list：生成约 {scene_count} 个镜头，每个镜头 {req.scene_seconds} 秒左右，必须包含人物、场景、机位、动作、光线、对白/旁白、音效。
2. 先生成角色/场景/产品参考：references.character、references.location、references.product、references.style 要能交给图片模型生成参考图，用来锁定角色、物体、地点、视觉风格一致性。
3. 逐镜头生成短视频：每个 scene 必须有 video_prompt，写成可直接交给图生视频/文生视频模型的长提示词，不是关键词。
4. 用首尾帧保持连贯：每个 scene 必须有 first_frame_prompt、last_frame_prompt、continuity。后一个镜头要承接前一个镜头的尾帧人物位置、动作方向、光线色温、道具和情绪。
5. 声音另做或同步生成：每个 scene 的 audio 写清楚环境声、对白/旁白、音效；默认旁白/字幕/配乐可后期添加。
6. 后期再统一：post_production 写清楚调色、转场、配乐、字幕、口型修正、插帧、超分、去闪烁、修手修脸等工作。

画面限制：
- 趋近真人实拍：真实办公室/工厂/门店/会议室，真实人物自然表情，真实光线，轻微手持或稳定器运镜。
- 避免一眼 AI：不要塑料皮肤、过度磨皮、夸张景深、赛博背景、漂浮图标、乱码文字、水印、畸形手指、假人感、角色漂移。
- 画面描述不要要求模型生成可读中文文字，字幕由后期添加。

只返回合法 JSON 对象，不要 Markdown，不要解释。字段结构如下：
{{
  "title": "长视频标题",
  "hook": "开场 15 秒钩子",
  "pipeline_summary": "一句话说明：短片段生成 + 分镜拼接 + 一致性控制 + 后期剪辑",
  "references": {{
    "character": "角色参考图提示词，锁定主角五官、年龄、发型、服装、气质",
    "location": "场景参考图提示词，锁定空间、光线、道具、色调",
    "product": "产品参考图提示词，锁定产品外观/界面/使用方式",
    "style": "全片摄影风格提示词，镜头语言、光线、色彩、质感"
  }},
  "outline": ["章节大纲"],
  "chapters": [
    {{
      "chapter": 1,
      "title": "章节标题",
      "goal": "本章目的",
      "scenes": [
        {{
          "scene": 1,
          "start": "00:00",
          "end": "00:20",
          "duration": 20,
          "title": "镜头标题",
          "narration": "完整旁白",
          "visual": "画面描述",
          "camera": "镜头/运镜",
          "subtitle": "短字幕",
          "first_frame_prompt": "本镜头首帧提示词",
          "last_frame_prompt": "本镜头尾帧提示词",
          "video_prompt": "本镜头短视频生成提示词，必须足够详细",
          "audio": "环境声/对白/旁白/音效",
          "transition": "和下一镜头的剪辑/转场方式",
          "continuity": "和前后镜头保持连续的角色、动作、光线、道具说明",
          "edit_notes": "后期处理建议"
        }}
      ]
    }}
  ],
  "shot_list": [
    "可选：如果不想重复，可以留空；系统会从 chapters.scenes 汇总"
  ],
  "post_production": ["调色", "补转场", "配乐", "字幕", "口型修正", "插帧", "超分", "去闪烁", "修手修脸"],
  "call_to_action": "结尾转化话术"
}}"""

    payload = {
        "model": settings.text_model,
        "messages": [
            {"role": "system", "content": "你是专业的企业长视频导演和获客脚本策划，始终返回合法 JSON 对象。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.65,
        "max_tokens": 12000,
    }
    headers = {"Authorization": f"Bearer {settings.text_api_key}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=180) as client:
        resp = await client.post(str(settings.text_api_url), json=payload, headers=headers)
    if resp.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"Long video script failed: {resp.text[:300]}")

    raw = resp.json()["choices"][0]["message"]["content"].strip()
    data = _parse_json_object(raw)
    return _normalize_long_video_script(data, req, topic)


def build_image_prompts(plan: CopyPlan, req: WorkflowRequest) -> list[str]:
    prompts = [
        _reference_image_prompt(
            "character",
            str(_first_storyboard_value(plan, "characterReferencePrompt") or plan.coverPrompt or plan.videoPrompt),
            req,
        ),
        _reference_image_prompt(
            "location",
            str(_first_storyboard_value(plan, "locationReferencePrompt") or plan.coverPrompt or plan.videoPrompt),
            req,
        ),
        _reference_image_prompt(
            "product",
            str(_first_storyboard_value(plan, "productReferencePrompt") or plan.coverPrompt or plan.videoPrompt),
            req,
        ),
    ]
    storyboard_prompts = [
        str(item.get("firstFramePrompt") or item.get("videoPrompt") or item.get("shot") or "")
        for item in plan.storyboard
    ]
    prompts = [*prompts, *storyboard_prompts]
    prompts = [p for p in prompts if p]
    if not prompts:
        prompts = [f"{req.brand_name or '品牌'}{req.platform}平台配图，{req.niche or req.topic}相关，{req.tone}风格"]
    return [_image_reference_realism_prompt(p, req) for p in prompts]


def _first_storyboard_value(plan: CopyPlan, key: str) -> str:
    for item in plan.storyboard:
        value = item.get(key) if isinstance(item, dict) else None
        if value:
            return str(value)
    return ""


def _reference_image_prompt(kind: str, prompt: str, req: WorkflowRequest) -> str:
    labels = {
        "character": "角色参考图：固定主角五官、年龄、发型、服装、职业气质和表情基调",
        "location": "场景参考图：固定办公室/会议室/门店/工厂等空间结构、光线、色调和道具位置",
        "product": "产品参考图：固定产品外观、界面、设备、使用方式和品牌气质",
    }
    return f"{labels.get(kind, '参考图')}。{prompt}。项目：{req.topic}"


def _image_reference_realism_prompt(prompt: str, req: WorkflowRequest) -> str:
    return (
        f"{prompt.strip()}。"
        f"{req.platform}视频参考图，{req.tone}风格。"
        "必须像真实摄影参考图，不是插画，不是海报，不是 3D 渲染。"
        "使用真实商业纪录片摄影：自然光、真实办公室/真实场景、真实人物肤色、正常手指和五官、细节可信。"
        "构图清晰，主体完整，预留后续图生视频运动空间。"
        "不要文字、不要水印、不要 Logo 乱字、不要塑料皮肤、不要过度磨皮、不要夸张景深、不要赛博背景。"
    )


def build_video_prompt(plan: CopyPlan, req: WorkflowRequest) -> str:
    scenes = []
    for idx, item in enumerate(plan.storyboard, start=1):
        label = item.get("duration") or f"镜头{item.get('scene') or idx}"
        scenes.append(
            " / ".join(
                part
                for part in [
                    f"{label}",
                    f"画面：{item.get('videoPrompt') or item.get('shot') or ''}",
                    f"机位：{item.get('camera') or '自然运镜'}",
                    f"首帧：{item.get('firstFramePrompt') or ''}",
                    f"尾帧：{item.get('lastFramePrompt') or ''}",
                    f"声音：{item.get('audio') or item.get('sound') or '环境声自然，旁白后期添加'}",
                    f"转场：{item.get('transition') or '动作承接剪辑'}",
                ]
                if part and not part.endswith("：")
            )
        )
    if scenes:
        return _human_realism_prompt(
            f"生成一个{req.video_duration}秒{req.platform}短视频，账号/品牌：{req.brand_name or '品牌官号'}。"
            f"主题：{plan.title}。整体风格：{req.tone}，{req.content_type}，{req.niche or ''}领域。"
            f"分镜 shot list：{' || '.join(scenes)}。"
            "要求：每个镜头 4-10 秒，画面连贯，主体一致，承接上一镜头尾帧的人物位置、动作方向、道具和光线。"
            "视频片段之间要像同一支真实拍摄素材，不是互不相关的 AI 片段。不要生成乱码文字，不要水印。"
        )
    return _human_realism_prompt(f"{req.brand_name or '品牌'}在{req.platform}的短视频。主题：{plan.title}。风格：{req.tone}。不要生成乱码文字，不要水印。")


async def _dashscope_images(settings: Settings, req: WorkflowRequest, prompts: list[str]) -> list[Asset]:
    if not settings.dashscope_api_key:
        raise HTTPException(status_code=400, detail="DASHSCOPE_API_KEY is not configured")
    headers = {
        "Authorization": f"Bearer {settings.dashscope_api_key}",
        "X-DashScope-Async": "enable",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=120) as client:
        task_ids = []
        for i in range(req.image_count):
            payload = {
                "model": settings.image_model,
                "input": {"prompt": prompts[i % len(prompts)]},
                "parameters": {"size": settings.image_size, "n": 1},
            }
            data = await _dashscope_submit_with_retry(client, DASHSCOPE_IMAGE_URL, payload, headers)
            task_id = data.get("output", {}).get("task_id")
            if not task_id:
                raise HTTPException(status_code=502, detail=f"Agent2 submit failed: {str(data)[:300]}")
            task_ids.append(task_id)
            await asyncio.sleep(1.5)

        images = []
        for idx, task_id in enumerate(task_ids, start=1):
            output = await _dashscope_poll(client, settings.dashscope_api_key, task_id)
            url = (output.get("results") or [{}])[0].get("url")
            if url:
                images.append(Asset(url=url, label=f"生成图{idx}"))
        return images


async def _qwen_images(settings: Settings, req: WorkflowRequest, prompts: list[str]) -> list[Asset]:
    if not settings.dashscope_api_key:
        raise HTTPException(status_code=400, detail="DASHSCOPE_API_KEY is not configured")
    headers = {
        "Authorization": f"Bearer {settings.dashscope_api_key}",
        "Content-Type": "application/json",
    }
    prompt = prompts[0]
    payload = {
        "model": settings.image_model,
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [{"text": prompt}],
                }
            ]
        },
        "parameters": {
            "size": settings.image_size,
            "n": req.image_count,
            "prompt_extend": True,
            "watermark": False,
            "negative_prompt": "低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，文字模糊，水印，乱码文字。",
        },
    }
    async with httpx.AsyncClient(timeout=180) as client:
        resp = await client.post(DASHSCOPE_QWEN_IMAGE_URL, json=payload, headers=headers)
    try:
        data = resp.json()
    except Exception:
        data = {"status_code": resp.status_code, "text": resp.text[:300]}
    if resp.status_code >= 400 or data.get("code"):
        raise HTTPException(status_code=502, detail=f"Qwen-Image failed: {str(data)[:500]}")

    urls = _extract_qwen_image_urls(data)
    if not urls:
        raise HTTPException(status_code=502, detail=f"Qwen-Image returned no image URL: {str(data)[:500]}")
    return [Asset(url=url, label=f"千问生成图{idx + 1}") for idx, url in enumerate(urls)]


def _extract_qwen_image_urls(data: dict) -> list[str]:
    urls: list[str] = []
    choices = data.get("output", {}).get("choices") or []
    for choice in choices:
        content = choice.get("message", {}).get("content") or []
        for item in content:
            url = item.get("image") or item.get("url")
            if isinstance(url, str):
                urls.append(url)
    results = data.get("output", {}).get("results") or []
    for item in results:
        url = item.get("url") if isinstance(item, dict) else None
        if isinstance(url, str):
            urls.append(url)
    return urls


async def _dashscope_submit_with_retry(
    client: httpx.AsyncClient,
    url: str,
    payload: dict,
    headers: dict,
    attempts: int = 3,
) -> dict:
    last_data: dict = {}
    for idx in range(attempts):
        resp = await client.post(url, json=payload, headers=headers)
        try:
            data = resp.json()
        except Exception:
            data = {"status_code": resp.status_code, "text": resp.text[:300]}
        last_data = data
        code = str(data.get("code") or "")
        message = str(data.get("message") or "")
        if resp.status_code < 400 and "RateQuota" not in code and "rate limit" not in message.lower():
            return data
        if "RateQuota" not in code and "rate limit" not in message.lower():
            return data
        await asyncio.sleep(4 * (idx + 1))
    return last_data


async def _dashscope_video(settings: Settings, prompt: str) -> str | None:
    if not settings.dashscope_api_key:
        raise HTTPException(status_code=400, detail="DASHSCOPE_API_KEY is not configured")
    params: dict[str, Any] = {"prompt": prompt}
    if settings.video_model.startswith("kling/"):
        params = {"prompt": prompt, "duration": settings.video_duration, "mode": "std", "aspect_ratio": "16:9"}
    headers = {
        "Authorization": f"Bearer {settings.dashscope_api_key}",
        "X-DashScope-Async": "enable",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(DASHSCOPE_VIDEO_URL, json={"model": settings.video_model, "input": params}, headers=headers)
        data = resp.json()
        task_id = data.get("output", {}).get("task_id")
        if not task_id:
            raise HTTPException(status_code=502, detail=f"Agent3 submit failed: {str(data)[:300]}")
        output = await _dashscope_poll(client, settings.dashscope_api_key, task_id, max_attempts=90, delay=2)
    result = (output.get("results") or [{}])[0]
    return result.get("video_url") or result.get("url") or output.get("video_url")


async def _qwen_image_to_video(settings: Settings, req: WorkflowRequest, prompt: str, images: list[Asset]) -> list[Asset]:
    if not settings.dashscope_api_key:
        raise HTTPException(status_code=400, detail="DASHSCOPE_API_KEY is not configured")
    if not images:
        raise HTTPException(status_code=400, detail="Qwen image-to-video requires at least one generated image")
    headers = {
        "Authorization": f"Bearer {settings.dashscope_api_key}",
        "X-DashScope-Async": "enable",
        "Content-Type": "application/json",
    }
    remaining = req.video_duration
    segment_index = 1
    videos: list[Asset] = []
    async with httpx.AsyncClient(timeout=120) as client:
        while remaining > 0:
            clip_duration = min(15, remaining)
            image = images[(segment_index - 1) % len(images)]
            segment_prompt = (
                f"{prompt}\n"
                f"这是总时长约 {req.video_duration} 秒视频的第 {segment_index} 个片段，本段时长 {clip_duration} 秒。"
                f"当前输入图片是本段 first frame/reference frame，必须严格保持图片中的角色、服装、地点、产品、道具和光线。"
                "如果不是第一段，请承接上一段尾帧的动作方向和情绪节奏，避免突然换人、换地点、换服装。"
                "镜头运动要克制真实：轻微手持、稳定器平移、自然推拉，不能大幅跳切或漂浮。"
                "输出应像真实拍摄素材，方便后期剪辑拼接；不要字幕文字、不要水印、不要 AI 塑料感。"
            )
            payload = {
                "model": settings.video_model,
                "input": {
                    "prompt": segment_prompt,
                    "media": [{"type": "first_frame", "url": image.url}],
                },
                "parameters": {
                    "resolution": "720P",
                    "duration": clip_duration,
                    "prompt_extend": True,
                    "watermark": False,
                },
            }
            data = await _dashscope_submit_with_retry(client, DASHSCOPE_VIDEO_URL, payload, headers, attempts=2)
            task_id = data.get("output", {}).get("task_id")
            if not task_id:
                raise HTTPException(status_code=502, detail=f"Qwen image-to-video submit failed: {str(data)[:500]}")
            output = await _dashscope_poll(client, settings.dashscope_api_key, task_id, max_attempts=120, delay=5)
            result = (output.get("results") or [{}])[0]
            url = result.get("video_url") or result.get("url") or output.get("video_url")
            if url:
                videos.append(Asset(url=url, label=f"千问图生视频片段{segment_index}（{clip_duration}秒）"))
            remaining -= clip_duration
            segment_index += 1
    return videos


async def concatenate_videos(videos: list[Asset], output_dir: Path) -> Asset:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        try:
            import imageio_ffmpeg

            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg = None
    if not ffmpeg:
        raise RuntimeError("本机未检测到 ffmpeg，无法把片段合成为一个 mp4。请安装 imageio-ffmpeg 或系统 ffmpeg 后重试。")

    job_id = uuid.uuid4().hex[:12]
    work_dir = output_dir / f"job-{job_id}"
    work_dir.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient(timeout=180, follow_redirects=True) as client:
        clip_paths = []
        for idx, video in enumerate(videos, start=1):
            resp = await client.get(video.url, headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code >= 400:
                raise RuntimeError(f"下载视频片段 {idx} 失败：HTTP {resp.status_code}")
            clip_path = work_dir / f"clip-{idx:03d}.mp4"
            clip_path.write_bytes(resp.content)
            clip_paths.append(clip_path)

    concat_file = work_dir / "concat.txt"
    concat_file.write_text(
        "\n".join(f"file '{path.as_posix()}'" for path in clip_paths),
        encoding="utf-8",
    )
    output_path = output_dir / f"combined-{job_id}.mp4"

    copy_cmd = [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(concat_file), "-c", "copy", str(output_path)]
    proc = subprocess.run(copy_cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        encode_cmd = [
            ffmpeg,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        proc = subprocess.run(encode_cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "unknown ffmpeg error")[-500:]
        raise RuntimeError(f"ffmpeg 合成失败：{detail}")

    return Asset(url=f"/generated/{output_path.name}", label="合成后完整视频")


async def _dashscope_poll(client: httpx.AsyncClient, api_key: str, task_id: str, max_attempts: int = 60, delay: int = 2) -> dict:
    headers = {"Authorization": f"Bearer {api_key}"}
    for _ in range(max_attempts):
        await asyncio.sleep(delay)
        resp = await client.get(DASHSCOPE_TASK_URL.format(task_id=task_id), headers=headers)
        data = resp.json()
        output = data.get("output", {})
        status = output.get("task_status")
        if status == "SUCCEEDED":
            return output
        if status == "FAILED":
            raise HTTPException(status_code=502, detail=f"DashScope task failed: {output.get('message') or output}")
    raise HTTPException(status_code=504, detail="DashScope task timeout")


async def _jimeng_images(settings: Settings, req: WorkflowRequest, prompts: list[str]) -> list[Asset]:
    images = []
    size = settings.jimeng_image_size
    width, height = (size.split("x") if "x" in size else (size, size))
    async with httpx.AsyncClient(timeout=120) as client:
        for idx in range(req.image_count):
            task_id = await _jimeng_submit(client, settings, settings.jimeng_image_model, prompts[idx % len(prompts)], int(width), int(height))
            url = await _jimeng_poll(client, settings, settings.jimeng_image_model, task_id)
            images.append(Asset(url=url, label=f"即梦生成图{idx + 1}"))
    return images


async def _jimeng_video(settings: Settings, prompt: str) -> str | None:
    async with httpx.AsyncClient(timeout=120) as client:
        task_id = await _jimeng_submit(client, settings, settings.jimeng_video_model, prompt, 1024, 576)
        return await _jimeng_poll(client, settings, settings.jimeng_video_model, task_id, max_attempts=90, delay=5)


async def _jimeng_submit(client: httpx.AsyncClient, settings: Settings, model: str, prompt: str, width: int, height: int) -> str:
    if not settings.jimeng_api_key:
        raise HTTPException(status_code=400, detail="JIMENG_API_KEY is not configured")
    is_video = "t2v" in model or "video" in model
    body = {"req_key": model, "prompt": prompt, "audio": True} if is_video else {"req_key": model, "prompt": prompt, "width": width, "height": height}
    resp = await client.post(
        JIMENG_URL,
        params={"Action": "CVSync2AsyncSubmitTask", "Version": "2022-08-31"},
        json=body,
        headers={"Authorization": settings.jimeng_api_key},
    )
    data = resp.json()
    task_id = data.get("data", {}).get("task_id") or data.get("task_id") or data.get("data", {}).get("TaskId")
    if not task_id:
        raise HTTPException(status_code=502, detail=f"Jimeng submit failed: {str(data)[:300]}")
    return task_id


async def _jimeng_poll(client: httpx.AsyncClient, settings: Settings, model: str, task_id: str, max_attempts: int = 90, delay: int = 2) -> str:
    for _ in range(max_attempts):
        await asyncio.sleep(delay)
        resp = await client.post(
            JIMENG_URL,
            params={"Action": "CVSync2AsyncGetResult", "Version": "2022-08-31"},
            json={"req_key": model, "task_id": task_id},
            headers={"Authorization": settings.jimeng_api_key},
        )
        data = resp.json()
        dd = data.get("data") or data
        nested = dd.get("data") or dd if isinstance(dd, dict) else {}
        if data.get("code") in (10000, "10000", 0, "0"):
            b64 = _first(dd, "binary_data_base64") or _first(nested, "binary_data_base64")
            if b64:
                return "data:image/png;base64," + b64
            url = (
                dd.get("video_url") or nested.get("video_url") or dd.get("image_url") or nested.get("image_url")
                or _first(dd, "results", "video_url") or _first(nested, "results", "video_url")
                or _first(dd, "results", "url") or _first(nested, "results", "url")
            )
            if isinstance(url, str) and (url.startswith("http") or url.startswith("data:")):
                return url
        if data.get("code") in (50501, 50000):
            raise HTTPException(status_code=502, detail=f"Jimeng task failed: {data.get('message')}")
    raise HTTPException(status_code=504, detail="Jimeng task timeout")


def _first(data: dict, key: str, child_key: str | None = None):
    value = data.get(key) if isinstance(data, dict) else None
    if isinstance(value, list) and value:
        first = value[0]
        if child_key and isinstance(first, dict):
            return first.get(child_key)
        return first
    return None


def _parse_json_array(raw: str) -> list[dict]:
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if match:
        raw = match.group(1).strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        array_match = re.search(r"\[\s*\{[\s\S]*\}\s*\]", raw)
        if not array_match:
            raise HTTPException(status_code=502, detail="Agent1 returned invalid JSON")
        data = json.loads(array_match.group(0))
    return data if isinstance(data, list) else [data]


def _parse_json_object(raw: str) -> dict:
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if match:
        raw = match.group(1).strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        object_match = re.search(r"\{\s*[\s\S]*\}\s*", raw)
        if not object_match:
            raise HTTPException(status_code=502, detail="Long video agent returned invalid JSON")
        data = json.loads(object_match.group(0))
    if not isinstance(data, dict):
        raise HTTPException(status_code=502, detail="Long video agent returned non-object JSON")
    return data


def _normalize_long_video_script(data: dict, req: LongVideoAgentRequest, topic: str) -> LongVideoScriptResult:
    chapters: list[LongVideoChapter] = []
    raw_chapters = data.get("chapters") if isinstance(data.get("chapters"), list) else []
    for chapter_idx, raw_chapter in enumerate(raw_chapters, start=1):
        if not isinstance(raw_chapter, dict):
            continue
        scenes: list[LongVideoScene] = []
        raw_scenes = raw_chapter.get("scenes") if isinstance(raw_chapter.get("scenes"), list) else []
        for scene_idx, raw_scene in enumerate(raw_scenes, start=1):
            if not isinstance(raw_scene, dict):
                continue
            scenes.append(
                LongVideoScene(
                    scene=int(raw_scene.get("scene") or scene_idx),
                    start=str(raw_scene.get("start") or ""),
                    end=str(raw_scene.get("end") or ""),
                    duration=max(4, min(10, int(raw_scene.get("duration") or req.scene_seconds))),
                    title=str(raw_scene.get("title") or f"镜头 {scene_idx}"),
                    narration=str(raw_scene.get("narration") or ""),
                    visual=str(raw_scene.get("visual") or raw_scene.get("videoPrompt") or ""),
                    camera=str(raw_scene.get("camera") or "自然运镜"),
                    subtitle=str(raw_scene.get("subtitle") or raw_scene.get("textOverlay") or ""),
                    first_frame_prompt=str(raw_scene.get("first_frame_prompt") or raw_scene.get("firstFramePrompt") or raw_scene.get("visual") or raw_scene.get("videoPrompt") or ""),
                    last_frame_prompt=str(raw_scene.get("last_frame_prompt") or raw_scene.get("lastFramePrompt") or raw_scene.get("visual") or raw_scene.get("videoPrompt") or ""),
                    video_prompt=str(raw_scene.get("video_prompt") or raw_scene.get("videoPrompt") or raw_scene.get("visual") or ""),
                    audio=str(raw_scene.get("audio") or raw_scene.get("sound") or "自然环境声，旁白/字幕/配乐后期添加"),
                    transition=str(raw_scene.get("transition") or "动作承接剪辑，必要时补转场"),
                    continuity=str(raw_scene.get("continuity") or raw_scene.get("continuity_notes") or "承接上一镜头尾帧的人物位置、动作方向、光线色温、道具状态和情绪节奏。"),
                    edit_notes=str(raw_scene.get("edit_notes") or raw_scene.get("post") or "统一调色，检查手部和脸部，必要时补帧去闪烁"),
                )
            )
        chapters.append(
            LongVideoChapter(
                chapter=int(raw_chapter.get("chapter") or chapter_idx),
                title=str(raw_chapter.get("title") or f"章节 {chapter_idx}"),
                goal=str(raw_chapter.get("goal") or ""),
                scenes=scenes,
            )
        )

    outline = data.get("outline") if isinstance(data.get("outline"), list) else []
    references = data.get("references") if isinstance(data.get("references"), dict) else {}
    post_production = data.get("post_production") if isinstance(data.get("post_production"), list) else []
    shot_list = [scene for chapter in chapters for scene in chapter.scenes]
    return LongVideoScriptResult(
        topic=topic,
        duration=req.video_duration,
        title=str(data.get("title") or f"{req.product_name or req.company_name or '企业服务'}长视频获客脚本"),
        hook=str(data.get("hook") or ""),
        pipeline_summary=str(
            data.get("pipeline_summary")
            or "短片段生成 + 分镜拼接 + 一致性控制 + 后期剪辑：模型生成可用镜头，系统保证镜头连续，剪辑让它像完整片子。"
        ),
        references=LongVideoReferenceSet(
            character=str(references.get("character") or data.get("characterReferencePrompt") or ""),
            location=str(references.get("location") or data.get("locationReferencePrompt") or ""),
            product=str(references.get("product") or data.get("productReferencePrompt") or ""),
            style=str(references.get("style") or data.get("styleReferencePrompt") or req.tone),
        ),
        outline=[str(item) for item in outline],
        chapters=chapters,
        shot_list=shot_list,
        post_production=[str(item) for item in post_production]
        or ["统一调色", "补转场", "配乐", "字幕", "口型修正", "插帧", "超分", "去闪烁", "修手修脸"],
        call_to_action=str(data.get("call_to_action") or data.get("cta") or ""),
    )


def _normalize_plan(item: dict, req: WorkflowRequest, idx: int) -> CopyPlan:
    caption = item.get("caption") or item.get("description") or ""
    keywords = item.get("keywords") if isinstance(item.get("keywords"), list) else []
    fallback = f"{item.get('title') or req.topic}，{caption[:80]}，{req.tone}风格，{req.platform}短视频，画面专业清晰"
    return CopyPlan(
        title=item.get("title") or f"{req.topic} 方案{idx + 1}",
        hook=item.get("hook") or item.get("subtitle") or "",
        subtitle=item.get("subtitle") or item.get("hook") or "",
        caption=caption,
        description=item.get("description") or caption,
        shortText=item.get("shortText") or " ".join(item.get("hashtags") or []),
        hashtags=item.get("hashtags") if isinstance(item.get("hashtags"), list) else [],
        keywords=keywords,
        coverPrompt=item.get("coverPrompt") or fallback,
        videoPrompt=item.get("videoPrompt") or fallback,
        storyboard=item.get("storyboard") if isinstance(item.get("storyboard"), list) else [],
    )

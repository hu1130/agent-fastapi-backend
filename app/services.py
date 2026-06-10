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
from .schemas import Asset, CopyPlan, WorkflowRequest


DASHSCOPE_IMAGE_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text2image/image-synthesis"
DASHSCOPE_QWEN_IMAGE_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
DASHSCOPE_VIDEO_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/video-generation/video-synthesis"
DASHSCOPE_TASK_URL = "https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}"
JIMENG_URL = "https://maas-openapi.wanjiedata.com/api/jimeng/v1"


async def generate_copywriting(settings: Settings, req: WorkflowRequest) -> list[CopyPlan]:
    if not settings.text_api_key:
        raise HTTPException(status_code=400, detail="TEXT_API_KEY is not configured")

    word_hint = {"短": "100字以内，精炼有力", "中": "200-400字，详略得当"}.get(req.word_limit, "500-800字，深度展开")
    scene_count = max(3, min(8, round(req.video_duration / 2)))
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


def build_image_prompts(plan: CopyPlan, req: WorkflowRequest) -> list[str]:
    storyboard_prompts = [str(item.get("videoPrompt") or item.get("shot") or "") for item in plan.storyboard]
    prompts = [plan.coverPrompt, *storyboard_prompts]
    prompts = [p for p in prompts if p]
    if not prompts:
        prompts = [f"{req.brand_name or '品牌'}{req.platform}平台配图，{req.niche or req.topic}相关，{req.tone}风格"]
    return [f"{p}。{req.platform}自媒体视觉，{req.tone}风格，画面清晰，构图完整，不要生成乱码文字，不要水印。" for p in prompts]


def build_video_prompt(plan: CopyPlan, req: WorkflowRequest) -> str:
    scenes = []
    for item in plan.storyboard:
        label = item.get("duration") or f"镜头{item.get('scene', '')}"
        scenes.append(f"{label}：{item.get('videoPrompt') or item.get('shot') or ''}；镜头：{item.get('camera') or '自然运镜'}")
    if scenes:
        return (
            f"生成一个{req.video_duration}秒{req.platform}短视频，账号/品牌：{req.brand_name or '品牌官号'}。"
            f"主题：{plan.title}。整体风格：{req.tone}，{req.content_type}，{req.niche or ''}领域。"
            f"分镜：{' / '.join(scenes)}。要求：画面连贯，主体一致，镜头自然，商业宣传片质感，不要生成乱码文字，不要水印。"
        )
    return f"{req.brand_name or '品牌'}在{req.platform}的短视频。主题：{plan.title}。风格：{req.tone}。不要生成乱码文字，不要水印。"


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
                f"这是总时长约 {req.video_duration} 秒长视频的第 {segment_index} 段，"
                f"本段时长 {clip_duration} 秒。保持同一视觉风格，画面承接上下文，适合后期拼接。"
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

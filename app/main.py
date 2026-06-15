from pathlib import Path
from urllib.parse import urlparse
from datetime import datetime, timezone
import json
import uuid

import httpx
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from .config import Settings, get_settings
from .customer_service import generate_customer_service_answer
from .schemas import BookingRequest, BookingResponse, CustomerServiceChatRequest, CustomerServiceChatResponse, CustomerServiceWidgetRequest, LongVideoAgentRequest, LongVideoTaskResponse, LongVideoTaskStatus, VideoAgentImageRequest, VideoAgentRequest, VideoAgentResponse, VideoAgentVideoRequest, VideoPromptResponse, WorkflowRequest, WorkflowResponse
from .services import build_image_prompts, build_video_prompt, concatenate_videos, generate_copywriting, generate_images, generate_images_from_prompts, generate_long_video_script, generate_video, generate_video_from_prompt


app = FastAPI(title="Agent Workflow API", version="1.0.0")

GENERATED_DIR = Path("generated")
GENERATED_DIR.mkdir(exist_ok=True)
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
BOOKINGS_FILE = DATA_DIR / "bookings.jsonl"


settings = get_settings()
LONG_VIDEO_TASKS: dict[str, LongVideoTaskStatus] = {}
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

app.mount("/static", StaticFiles(directory="frontend"), name="static")
app.mount("/generated", StaticFiles(directory=GENERATED_DIR), name="generated")


@app.get("/")
async def index():
    return FileResponse("frontend/index.html")


@app.get("/api/health")
async def health():
    return {"ok": True}


@app.get("/api/config/status")
async def config_status(cfg: Settings = Depends(get_settings)):
    return {
        "text": bool(cfg.text_api_key),
        "dashscope": bool(cfg.dashscope_api_key),
        "jimeng": bool(cfg.jimeng_api_key),
        "image_provider": cfg.image_provider,
        "video_provider": cfg.video_provider,
    }


@app.post("/api/bookings", response_model=BookingResponse)
async def create_booking(req: BookingRequest):
    booking_id = uuid.uuid4().hex
    created_at = datetime.now(timezone.utc).isoformat()
    record = {
        "booking_id": booking_id,
        "created_at": created_at,
        "company": req.company,
        "name": req.name,
        "phone": req.phone,
        "agents": req.agents,
        "source": req.source,
    }
    with BOOKINGS_FILE.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
    return BookingResponse(booking_id=booking_id, created_at=created_at)


@app.get("/api/bookings")
async def list_bookings():
    if not BOOKINGS_FILE.exists():
        return {"items": []}
    items = []
    with BOOKINGS_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return {"items": items}


@app.post("/api/customer-service/chat", response_model=CustomerServiceChatResponse)
async def customer_service_chat(req: CustomerServiceChatRequest, cfg: Settings = Depends(get_settings)):
    return await generate_customer_service_answer(cfg, req.message, req.session_id)


@app.post("/v1/chat", response_model=CustomerServiceChatResponse)
async def customer_service_widget_chat(req: CustomerServiceWidgetRequest, cfg: Settings = Depends(get_settings)):
    return await generate_customer_service_answer(cfg, req.message, req.session_id)


@app.post("/api/workflow/run", response_model=WorkflowResponse)
async def run_workflow(req: WorkflowRequest, cfg: Settings = Depends(get_settings)):
    return await _run_video_workflow(req, cfg)


@app.post("/api/video-agent/generate", response_model=VideoAgentResponse)
async def generate_video_agent(req: VideoAgentRequest, cfg: Settings = Depends(get_settings)):
    topic_parts = [
        f"公司/品牌：{req.company_name}" if req.company_name else "",
        f"行业：{req.industry}" if req.industry else "",
        f"产品/服务：{req.product_name}" if req.product_name else "",
        f"核心卖点：{req.selling_points}" if req.selling_points else "",
        f"获客目标：{req.goal}" if req.goal else "",
    ]
    topic = "；".join(part for part in topic_parts if part) or "企业AI智能体获客短视频"
    features = [item.strip() for item in req.selling_points.replace("，", ",").split(",") if item.strip()]
    if not features:
        features = ["企业流程简化", "AI智能体提效", "获客转化"]

    workflow_req = WorkflowRequest(
        brand_name=req.company_name,
        brand_tagline=req.goal,
        platform=req.platform,
        niche=req.industry,
        tone=req.tone,
        audience=req.target_customer,
        topic=topic,
        content_type="短视频",
        copy_count=req.copy_count,
        image_count=req.image_count,
        word_limit="中",
        features=features,
        extra_desc=req.extra_desc,
        video_duration=req.video_duration,
        generate_video=req.generate_video,
    )
    result = await _run_video_workflow(workflow_req, cfg)
    return VideoAgentResponse(topic=topic, **result.model_dump())


@app.post("/api/video-agent/prompts", response_model=VideoPromptResponse)
async def generate_video_prompts(req: VideoAgentRequest, cfg: Settings = Depends(get_settings)):
    workflow_req = _build_video_agent_workflow(req, image_count=0, generate_video=False)
    copywriting = await generate_copywriting(cfg, workflow_req)
    first = copywriting[0]
    return VideoPromptResponse(
        topic=workflow_req.topic,
        copywriting=copywriting,
        video_prompt=build_video_prompt(first, workflow_req),
        image_prompts=build_image_prompts(first, workflow_req),
    )


@app.post("/api/video-agent/images/regenerate", response_model=WorkflowResponse)
async def regenerate_video_images(req: VideoAgentImageRequest, cfg: Settings = Depends(get_settings)):
    workflow_req = _build_video_agent_workflow(req, image_count=req.image_count, generate_video=False)
    images = await generate_images_from_prompts(cfg, workflow_req, req.image_prompts)
    return WorkflowResponse(copywriting=[], images=images, videos=[], warnings=[])


@app.post("/api/video-agent/videos/regenerate", response_model=WorkflowResponse)
async def regenerate_video(req: VideoAgentVideoRequest, cfg: Settings = Depends(get_settings)):
    workflow_req = _build_video_agent_workflow(req, image_count=0, generate_video=True)
    videos = await generate_video_from_prompt(cfg, workflow_req, req.video_prompt, req.images)
    video_url = videos[0].url if videos else None
    warnings: list[str] = []
    if len(videos) > 1:
        try:
            combined = await concatenate_videos(videos, GENERATED_DIR)
            videos = [combined]
            video_url = combined.url
            warnings.append("已将多个 15 秒左右的视频片段自动拼接为一个 mp4。")
        except Exception as exc:
            warnings.append(f"已生成多个片段，但自动拼接失败：{exc}")
    return WorkflowResponse(copywriting=[], images=req.images, video_url=video_url, videos=videos, warnings=warnings)


@app.post("/api/long-video-agent/tasks", response_model=LongVideoTaskResponse)
async def create_long_video_task(
    req: LongVideoAgentRequest,
    background_tasks: BackgroundTasks,
    cfg: Settings = Depends(get_settings),
):
    task_id = uuid.uuid4().hex
    task = LongVideoTaskStatus(task_id=task_id, status="queued", message="长视频脚本任务已提交")
    LONG_VIDEO_TASKS[task_id] = task
    background_tasks.add_task(_run_long_video_task, task_id, req, cfg)
    return LongVideoTaskResponse(task_id=task_id, status=task.status, message=task.message)


@app.get("/api/long-video-agent/tasks/{task_id}", response_model=LongVideoTaskStatus)
async def get_long_video_task(task_id: str):
    task = LONG_VIDEO_TASKS.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Long video task not found")
    return task


async def _run_long_video_task(task_id: str, req: LongVideoAgentRequest, cfg: Settings):
    task = LONG_VIDEO_TASKS[task_id]
    task.status = "running"
    task.message = "正在生成视频脚本和分镜"
    try:
        result = await generate_long_video_script(cfg, req)
        task.status = "succeeded"
        task.message = "长视频脚本和分镜已生成"
        task.result = result
    except Exception as exc:
        task.status = "failed"
        task.message = "长视频脚本生成失败"
        task.error = str(exc)


def _build_video_agent_workflow(
    req: VideoAgentRequest,
    image_count: int | None = None,
    generate_video: bool | None = None,
) -> WorkflowRequest:
    topic_parts = [
        f"公司/品牌：{req.company_name}" if req.company_name else "",
        f"行业：{req.industry}" if req.industry else "",
        f"产品/服务：{req.product_name}" if req.product_name else "",
        f"核心卖点：{req.selling_points}" if req.selling_points else "",
        f"获客目标：{req.goal}" if req.goal else "",
    ]
    topic = "；".join(part for part in topic_parts if part) or "企业 AI 智能体获客视频"
    feature_text = req.selling_points.replace("；", ",").replace("、", ",")
    features = [item.strip() for item in feature_text.split(",") if item.strip()]
    if not features:
        features = ["企业流程简化", "AI 智能体提效", "获客转化"]
    return WorkflowRequest(
        brand_name=req.company_name,
        brand_tagline=req.goal,
        platform=req.platform,
        niche=req.industry,
        tone=req.tone,
        audience=req.target_customer,
        topic=topic,
        content_type="获客视频",
        copy_count=req.copy_count,
        image_count=image_count if image_count is not None else req.image_count,
        word_limit="长",
        features=features,
        extra_desc=req.extra_desc,
        video_duration=req.video_duration,
        generate_video=req.generate_video if generate_video is None else generate_video,
    )


async def _run_video_workflow(req: WorkflowRequest, cfg: Settings) -> WorkflowResponse:
    warnings: list[str] = []
    copywriting = await generate_copywriting(cfg, req)

    images = []
    try:
        images = await generate_images(cfg, req, copywriting)
    except Exception as exc:
        warnings.append(f"Agent2 image generation failed: {exc}")

    videos = []
    video_url = None
    if req.generate_video:
        try:
            videos = await generate_video(cfg, req, copywriting, images)
            video_url = videos[0].url if videos else None
        except Exception as exc:
            warnings.append(f"Agent3 video generation failed: {exc}")

    if len(videos) > 1:
        try:
            combined = await concatenate_videos(videos, GENERATED_DIR)
            videos = [combined]
            video_url = combined.url
            warnings.append("长视频已由多个短片段自动合成为一个 mp4。")
        except Exception as exc:
            warnings.append(f"已生成多个短片段，但自动合成失败：{exc}")

    return WorkflowResponse(copywriting=copywriting, images=images, video_url=video_url, videos=videos, warnings=warnings)


@app.get("/api/video-proxy")
async def video_proxy(url: str = Query(..., min_length=8)):
    parsed = urlparse(url)
    allowed_hosts = {
        "dashscope-result-sh.oss-cn-shanghai.aliyuncs.com",
        "dashscope-result-bj.oss-cn-beijing.aliyuncs.com",
        "v3-default.ixigua.com",
        "p3-sign.douyinpic.com",
        "p6-sign.douyinpic.com",
    }
    host = parsed.hostname or ""
    allowed_oss_host = host.endswith(".oss-accelerate.aliyuncs.com") or (
        host.startswith("dashscope-") and ".oss-" in host and host.endswith(".aliyuncs.com")
    )
    if parsed.scheme not in {"http", "https"} or (host not in allowed_hosts and not allowed_oss_host):
        raise HTTPException(status_code=400, detail="Video host is not allowed")

    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://jimeng.jianying.com/"}
    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
        resp = await client.get(url, headers=headers)
    if resp.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"Video fetch failed: {resp.status_code}")
    return Response(
        content=resp.content,
        media_type=resp.headers.get("content-type", "video/mp4"),
        headers={"Cache-Control": "public, max-age=3600"},
    )

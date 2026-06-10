from pathlib import Path
from urllib.parse import urlparse

import httpx
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from .config import Settings, get_settings
from .schemas import VideoAgentRequest, VideoAgentResponse, WorkflowRequest, WorkflowResponse
from .services import concatenate_videos, generate_copywriting, generate_images, generate_video


app = FastAPI(title="Agent Workflow API", version="1.0.0")

GENERATED_DIR = Path("generated")
GENERATED_DIR.mkdir(exist_ok=True)


settings = get_settings()
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

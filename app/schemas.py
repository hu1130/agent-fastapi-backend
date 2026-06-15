from pydantic import BaseModel, Field, field_validator


class WorkflowRequest(BaseModel):
    brand_name: str = ""
    brand_tagline: str = ""
    platform: str = "抖音"
    niche: str = ""
    tone: str = "专业可信"
    audience: str = "企业客户"
    reference: str = ""

    topic: str = Field(..., min_length=1, max_length=5000)
    content_type: str = "短视频"
    copy_count: int = Field(3, ge=1, le=5)
    image_count: int = Field(3, ge=0, le=4)
    word_limit: str = "中"
    features: list[str] = Field(default_factory=lambda: ["实用干货"])
    extra_desc: str = ""

    image_provider: str | None = None
    video_provider: str | None = None
    video_duration: int = Field(30, ge=30, le=600)
    generate_video: bool = True

    @field_validator("video_duration", mode="before")
    @classmethod
    def coerce_video_duration(cls, value):
        if value in ("", None):
            return 30
        return value

    @field_validator("copy_count", "image_count", mode="before")
    @classmethod
    def coerce_numbers(cls, value):
        if value in ("", None):
            return 5
        return value


class CopyPlan(BaseModel):
    title: str = ""
    hook: str = ""
    subtitle: str = ""
    caption: str = ""
    description: str = ""
    shortText: str = ""
    hashtags: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    coverPrompt: str = ""
    videoPrompt: str = ""
    storyboard: list[dict] = Field(default_factory=list)


class Asset(BaseModel):
    url: str
    label: str


class WorkflowResponse(BaseModel):
    copywriting: list[CopyPlan]
    images: list[Asset]
    video_url: str | None = None
    videos: list[Asset] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class VideoAgentRequest(BaseModel):
    company_name: str = ""
    industry: str = ""
    product_name: str = ""
    selling_points: str = ""
    target_customer: str = "企业客户"
    platform: str = "抖音"
    goal: str = "获客引流"
    tone: str = "专业可信"
    extra_desc: str = ""
    copy_count: int = Field(3, ge=1, le=5)
    image_count: int = Field(3, ge=0, le=4)
    video_duration: int = Field(30, ge=30, le=600)
    generate_video: bool = True


class VideoAgentResponse(WorkflowResponse):
    topic: str


class VideoPromptResponse(BaseModel):
    topic: str
    copywriting: list[CopyPlan]
    video_prompt: str = ""
    image_prompts: list[str] = Field(default_factory=list)


class VideoAgentImageRequest(VideoAgentRequest):
    image_prompts: list[str] = Field(default_factory=list)
    image_count: int = Field(2, ge=1, le=4)


class VideoAgentVideoRequest(VideoAgentRequest):
    video_prompt: str = Field(..., min_length=5)
    images: list[Asset] = Field(default_factory=list)
    video_duration: int = Field(30, ge=30, le=600)


class LongVideoAgentRequest(VideoAgentRequest):
    video_duration: int = Field(30, ge=30, le=600)
    chapter_count: int = Field(6, ge=4, le=12)
    scene_seconds: int = Field(8, ge=4, le=10)


class LongVideoReferenceSet(BaseModel):
    character: str = ""
    location: str = ""
    product: str = ""
    style: str = ""


class LongVideoScene(BaseModel):
    scene: int
    start: str = ""
    end: str = ""
    duration: int = 0
    title: str = ""
    narration: str = ""
    visual: str = ""
    camera: str = ""
    subtitle: str = ""
    first_frame_prompt: str = ""
    last_frame_prompt: str = ""
    video_prompt: str = ""
    audio: str = ""
    transition: str = ""
    continuity: str = ""
    edit_notes: str = ""


class LongVideoChapter(BaseModel):
    chapter: int
    title: str = ""
    goal: str = ""
    scenes: list[LongVideoScene] = Field(default_factory=list)


class LongVideoScriptResult(BaseModel):
    topic: str = ""
    duration: int = 30
    title: str = ""
    hook: str = ""
    pipeline_summary: str = ""
    references: LongVideoReferenceSet = Field(default_factory=LongVideoReferenceSet)
    outline: list[str] = Field(default_factory=list)
    chapters: list[LongVideoChapter] = Field(default_factory=list)
    shot_list: list[LongVideoScene] = Field(default_factory=list)
    post_production: list[str] = Field(default_factory=list)
    call_to_action: str = ""


class LongVideoTaskResponse(BaseModel):
    task_id: str
    status: str
    message: str = ""


class LongVideoTaskStatus(LongVideoTaskResponse):
    result: LongVideoScriptResult | None = None
    error: str | None = None


class BookingRequest(BaseModel):
    company: str = Field(..., min_length=1, max_length=120)
    name: str = Field(..., min_length=1, max_length=60)
    phone: str = Field(..., pattern=r"^1[3-9]\d{9}$")
    agents: list[str] = Field(default_factory=list)
    source: str = "website"


class BookingResponse(BaseModel):
    ok: bool = True
    booking_id: str
    created_at: str


class CustomerServiceChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: str = "website"
    api_key: str | None = None


class CustomerServiceWidgetRequest(BaseModel):
    api_key: str = ""
    session_id: str = "website"
    message: str = Field(..., min_length=1, max_length=2000)


class CustomerServiceMessage(BaseModel):
    text: str
    delay_ms: int = 0
    should_promote: bool = False


class CustomerServiceChatResponse(BaseModel):
    session_id: str
    intent: str = "chat"
    emotion: str = "neutral"
    audience: str = "general"
    answer: str = ""
    messages: list[CustomerServiceMessage] = Field(default_factory=list)

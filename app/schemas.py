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
    video_duration: int = Field(5, ge=3, le=180)
    generate_video: bool = True

    @field_validator("video_duration", "copy_count", "image_count", mode="before")
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
    video_duration: int = Field(5, ge=3, le=180)
    generate_video: bool = True


class VideoAgentResponse(WorkflowResponse):
    topic: str

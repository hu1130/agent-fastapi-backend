# FastAPI Agent 后端

企业智能体 SaaS 后端，包含：

- 原有 AI 内容生成工作流
- AI 短视频获客 Agent 接口
- 文案、图片、视频生成链路

## 本地启动

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
python -m uvicorn app.main:app --host 127.0.0.1 --port 8888
```

健康检查：

```text
GET http://127.0.0.1:8888/api/health
```

短视频获客 Agent：

```text
POST http://127.0.0.1:8888/api/video-agent/generate
```

长视频脚本/分镜 Agent（任务制，只生成生产方案，不直接生成真实视频）：

```text
POST http://127.0.0.1:8888/api/long-video-agent/tasks
GET  http://127.0.0.1:8888/api/long-video-agent/tasks/{task_id}
```

视频脚本时长范围：30-600 秒。后端不会一次性让模型吐出完整长视频，而是按市面上更常见的流水线组织：

1. LLM 先拆 shot list，每个镜头 4-10 秒。
2. 先生成角色、场景、产品、风格参考图提示词，用来控制一致性。
3. 每个镜头生成独立 video_prompt，并带 first_frame_prompt / last_frame_prompt。
4. 后一镜头承接前一镜头尾帧的人物位置、动作方向、道具、光线和情绪。
5. 声音可以单独后期，也可以交给支持原生音频的视频模型。
6. 最后通过调色、补转场、配乐、字幕、口型修正、插帧、超分、去闪烁、修手修脸统一成片。

一句话：模型负责生成“可用镜头”，系统负责让镜头连续，剪辑负责让它像片子。

任务状态保存在当前 FastAPI 进程内，重启服务后会清空。

分阶段视频生成接口：

```text
POST http://127.0.0.1:8888/api/video-agent/prompts
POST http://127.0.0.1:8888/api/video-agent/images/regenerate
POST http://127.0.0.1:8888/api/video-agent/videos/regenerate
```

推荐流程：先生成提示词，人工修改后生成/重生成图片，再用当前图片和当前视频提示词生成/重生成视频。视频会按约 15 秒片段生成，多个片段会自动拼接。

注意：不要提交真实 `.env` 或 API Key。

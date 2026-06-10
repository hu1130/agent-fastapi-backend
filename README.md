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

注意：不要提交真实 `.env` 或 API Key。

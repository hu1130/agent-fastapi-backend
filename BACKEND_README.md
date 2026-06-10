# 多 Agent 协同工作流 FastAPI 版

这是把原来的本地代理页面改造成的前后端接口版：

- 前端：`frontend/index.html`
- 后端：FastAPI，入口为 `app/main.py`
- 模型密钥：放在服务端 `.env`，不再由浏览器保存
- 外部接口：后端固定调用 DeepSeek、DashScope/百炼、即梦，不接受前端传任意目标 URL

## 运行

```powershell
cd outputs\agent-workflow-fastapi
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
python run.py
```

然后打开：

```text
http://127.0.0.1:8888
```

## 主要接口

- `GET /api/health`
- `GET /api/config/status`
- `POST /api/workflow/run`
- `GET /api/video-proxy?url=...`

## 云端部署前建议

- `APP_HOST` 本地用 `127.0.0.1`，云端容器内可用 `0.0.0.0`
- `ALLOWED_ORIGINS` 改成自己的域名
- 加登录鉴权、用户额度、IP 限流
- 生产环境不要开 `reload=True`
- 视频代理白名单按实际视频域名补充，不要放开任意 URL

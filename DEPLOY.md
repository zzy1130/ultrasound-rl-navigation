# 部署指南 / Deployment Guide

本项目可以通过多种方式部署到线上，让所有人都能访问。

## 方案一：Docker Compose（推荐）

### 前置要求
- Docker & Docker Compose
- 一台服务器（推荐 4GB+ 内存）

### 部署步骤

1. **克隆项目到服务器**
```bash
git clone <your-repo-url>
cd ultrasound-rl-navigation
```

2. **配置环境变量**
```bash
# 创建 .env 文件
echo "API_URL=http://your-server-ip:8765" > .env
```

3. **构建并启动**
```bash
docker-compose up -d --build
```

4. **访问应用**
- 前端：`http://your-server-ip:80`
- API：`http://your-server-ip:8765`

---

## 方案二：分离部署（前后端分开）

### 后端部署选项

#### Railway（免费额度）
```bash
# 安装 Railway CLI
npm install -g @railway/cli

# 登录并部署
railway login
railway init
railway up
```

#### Render
1. 连接 GitHub 仓库
2. 选择 Web Service
3. Build Command: `pip install uv && uv pip install --system -e .`
4. Start Command: `uvicorn api:app --host 0.0.0.0 --port $PORT`

#### Fly.io
```bash
# 安装 flyctl
curl -L https://fly.io/install.sh | sh

# 部署
fly launch
fly deploy
```

### 前端部署选项

#### Vercel（推荐，免费）
```bash
cd webapp
npm install -g vercel
vercel
```

在 Vercel 设置中添加环境变量：
- `VITE_API_URL` = 你的后端 API URL

#### Netlify
```bash
cd webapp
npm run build
# 上传 dist 文件夹到 Netlify
```

---

## 方案三：云服务器手动部署

### 1. 安装依赖
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3.10 python3-pip nodejs npm nginx

# 安装 uv
pip install uv
```

### 2. 部署后端
```bash
cd ultrasound-rl-navigation
uv pip install --system -e .
nohup uvicorn api:app --host 0.0.0.0 --port 8765 &
```

### 3. 部署前端
```bash
cd webapp
npm install
npm run build

# 复制到 nginx 目录
sudo cp -r dist/* /var/www/html/
```

### 4. 配置 Nginx
```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    root /var/www/html;
    index index.html;
    
    location / {
        try_files $uri $uri/ /index.html;
    }
    
    location /api {
        proxy_pass http://localhost:8765;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
    }
}
```

---

## 注意事项

### 模型文件
确保 `results/trained_models/` 目录包含：
- `simple_resnet_unet_best.pth`
- `agent_final.pt`

### 内存要求
- 最低：2GB RAM
- 推荐：4GB+ RAM（PyTorch 模型需要较多内存）

### HTTPS（生产环境）
使用 Let's Encrypt 获取免费 SSL 证书：
```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

---

## 推荐的免费部署组合

| 服务 | 用途 | 免费额度 |
|------|------|---------|
| Railway | 后端 API | 500小时/月 |
| Vercel | 前端托管 | 无限制 |
| Cloudflare | CDN + SSL | 无限制 |

这个组合可以让你零成本部署一个生产级别的应用！


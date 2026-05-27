# Rag-fullstack-docker-AWS Setup Guide

## Quick Setup

### 1. Clone Repository

```bash
git clone https://github.com/Amith-Ganta/Rag-fullstack-docker-AWS.git
cd Rag-fullstack-docker-AWS
```

### 2. Local Development Setup

#### Option A: Using Docker Compose (Recommended)

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your API keys
nano .env

# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

#### Option B: Manual Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
cp .env.example .env
nano .env

# Run API server
python -m uvicorn backend.main:app --reload

# In another terminal, run frontend
streamlit run frontend/app.py
```

### 3. Environment Variables

Create `.env` file with:

```env
# LLM Configuration
LLM_PROVIDER=groq
LLM_API_KEY=your_groq_api_key_here

# Alternative LLM providers
# LLM_PROVIDER=deepseek
# LLM_API_KEY=your_deepseek_key_here

# LLM_PROVIDER=openai
# LLM_API_KEY=your_openai_key_here

# Web Search
TAVILY_API_KEY=your_tavily_api_key_here

# Application Settings
ENVIRONMENT=development
DEBUG=true
API_URL=http://localhost:8000
```

### 4. Access Applications

- **API Documentation**: http://localhost:8000/docs
- **API**: http://localhost:8000
- **Streamlit Frontend**: http://localhost:8501

## GitHub Actions Setup

### 1. Create GitHub Repository

```bash
# Initialize git
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: RAG fullstack application with CI/CD"

# Add remote
git remote add origin https://github.com/Amith-Ganta/Rag-fullstack-docker-AWS.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### 2. Configure GitHub Secrets

Go to **Settings → Secrets and variables → Actions** and add:

| Secret | Value |
|--------|-------|
| `DOCKER_USERNAME` | Your Docker Hub username |
| `DOCKER_TOKEN` | Your Docker Hub access token |
| `AWS_ACCESS_KEY_ID` | Your AWS IAM access key |
| `AWS_SECRET_ACCESS_KEY` | Your AWS IAM secret key |
| `EC2_SSH_PRIVATE_KEY` | Contents of your `.pem` file (run: `cat ~/.ssh/your-key.pem`) |

### 3. Verify CI/CD Pipeline

After pushing to main:
1. Go to **Actions** tab on GitHub
2. Watch the pipeline run
3. Check status of each job (Build & Test, Build Docker, Deploy to EC2)

### 4. Get Docker Hub Access Token

1. Go to https://hub.docker.com/settings/security
2. Click "New Access Token"
3. Give it a name like "github-actions"
4. Copy the token to `DOCKER_TOKEN` secret

### 5. Get AWS Credentials

```bash
# List IAM users
aws iam list-users

# Create access key for your user
aws iam create-access-key --user-name your-username

# Save the credentials securely
```

### 6. EC2 Setup

```bash
# Create EC2 instance (if not already running)
aws ec2 run-instances \
  --image-id ami-0c55b159cbfafe1f0 \
  --instance-type t3.micro \
  --key-name your-key-pair \
  --security-groups default

# Install Docker on EC2
ssh -i your-key.pem ec2-user@<instance-ip>

# On EC2:
sudo yum update -y
sudo yum install docker -y
sudo usermod -aG docker ec2-user
logout

# Verify Docker works
ssh -i your-key.pem ec2-user@<instance-ip> docker --version
```

## Testing

### Run Local Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=backend --cov=frontend --cov-report=html

# Run specific test
pytest tests/test_api.py::TestHealthCheck -v
```

### Test API Manually

```bash
# Health check
curl http://localhost:8000/health

# Query endpoint
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG?", "model": "groq"}'

# Upload document
curl -X POST http://localhost:8000/documents/upload \
  -F "file=@document.pdf"

# List documents
curl http://localhost:8000/documents
```

## Deployment

### Manual Deployment to EC2

```bash
# Build Docker image
docker build -t rag-fullstack:latest .

# Tag for Docker Hub
docker tag rag-fullstack:latest amith98480/rag-fullstack-docker-aws:latest

# Login to Docker Hub
docker login

# Push to Docker Hub
docker push amith98480/rag-fullstack-docker-aws:latest

# SSH to EC2
ssh -i your-key.pem ec2-user@<instance-ip>

# On EC2: Pull and run
docker pull amith98480/rag-fullstack-docker-aws:latest
docker run -d \
  --name rag-fullstack \
  -p 8000:8000 \
  -p 8501:8501 \
  --restart unless-stopped \
  amith98480/rag-fullstack-docker-aws:latest
```

### Automatic Deployment (via GitHub Actions)

1. Commit and push to `main` branch
2. GitHub Actions automatically:
   - Builds and tests code
   - Builds Docker image
   - Pushes to Docker Hub
   - Deploys to EC2
3. Check progress in GitHub Actions tab

## Troubleshooting

### API not starting

```bash
# Check logs
docker logs rag-fullstack

# Check if port is in use
sudo lsof -i :8000

# Kill process on port
sudo kill -9 <PID>
```

### Docker build fails

```bash
# Clear Docker cache
docker builder prune -a

# Build with no cache
docker build --no-cache -t rag-fullstack:latest .
```

### EC2 deployment fails

```bash
# Check EC2 instance status
aws ec2 describe-instances --instance-ids i-0f790c856f47094cb

# SSH to EC2 and check Docker
ssh -i your-key.pem ec2-user@<instance-ip>
docker ps
docker logs rag-fullstack
```

### GitHub Actions fails

1. Check **Actions** tab for error logs
2. Verify all secrets are set correctly
3. Check EC2 security group allows SSH (port 22)
4. Verify `.pem` file has correct permissions

## Project Structure

```
Rag-fullstack-docker-AWS/
├── .github/
│   └── workflows/
│       └── ci-cd.yml              # GitHub Actions pipeline
├── backend/
│   ├── main.py                    # FastAPI application
│   └── __init__.py
├── frontend/
│   ├── app.py                     # Streamlit application
│   └── __init__.py
├── tests/
│   ├── test_api.py                # API tests
│   └── __init__.py
├── .env.example                   # Environment template
├── .gitignore                     # Git ignore rules
├── Dockerfile                     # Multi-stage Docker build
├── docker-compose.yml             # Local development
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
├── SETUP.md                       # Setup guide (this file)
└── .dockerignore                  # Docker ignore rules
```

## Getting Help

- 📚 [FastAPI Docs](https://fastapi.tiangolo.com/)
- 🎈 [Streamlit Docs](https://docs.streamlit.io/)
- 🐳 [Docker Docs](https://docs.docker.com/)
- ☁️ [AWS EC2 Docs](https://docs.aws.amazon.com/ec2/)
- 🔄 [GitHub Actions](https://docs.github.com/en/actions)

## Next Steps

1. ✅ Clone/setup repository
2. ✅ Configure environment variables
3. ✅ Test locally with Docker Compose
4. ✅ Push to GitHub
5. ✅ Configure GitHub secrets
6. ✅ Trigger deployment via GitHub Actions
7. ✅ Access running application on EC2

Enjoy your RAG fullstack application! 🚀

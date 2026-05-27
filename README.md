# RAG Fullstack Docker AWS

A production-ready RAG (Retrieval-Augmented Generation) application with FastAPI backend, Streamlit frontend, Docker containerization, and automated AWS EC2 deployment via GitHub Actions CI/CD.

## Features

- **FastAPI Backend** - RESTful API with automatic documentation
- **Streamlit Frontend** - Interactive web UI for RAG interactions
- **LLM Integration** - Support for multiple LLMs (Groq, DeepSeek, OpenAI)
- **Document Retrieval** - Local document indexing and retrieval
- **Web Search** - Tavily integration for web-based answers
- **Docker Support** - Multi-container setup with docker-compose
- **AWS Deployment** - Automated CI/CD pipeline to EC2 via GitHub Actions
- **Health Checks** - Endpoint monitoring and auto-restart

## Quick Start

### Local Development

```bash
# Clone repository
git clone https://github.com/Amith-Ganta/Rag-fullstack-docker-AWS.git
cd Rag-fullstack-docker-AWS

# Install dependencies
pip install -r requirements.txt

# Run with docker-compose
docker-compose up -d

# Access applications
# API: http://localhost:8000/docs
# Frontend: http://localhost:8501
```

### Environment Variables

Create a `.env` file in the project root:

```env
# LLM Configuration
LLM_PROVIDER=groq  # groq, deepseek, openai
LLM_API_KEY=your_api_key_here

# Tavily Web Search
TAVILY_API_KEY=your_tavily_key_here

# Application
ENVIRONMENT=production
DEBUG=false
```

## CI/CD Pipeline

GitHub Actions automatically:

1. **Build & Test** - Run linting, unit tests, and code coverage
2. **Build Docker** - Create Docker image from Dockerfile and push to Docker Hub
3. **Deploy to EC2** - SSH into EC2 instance and deploy latest container

### Triggering Deployments

- **Automatic**: Push to `main` or `develop` branch triggers the pipeline
- **Manual**: Use GitHub Actions "Run workflow" to trigger manually
- **PR**: Pull requests run build & test only (no Docker push or EC2 deployment)

### GitHub Secrets Required

Add these secrets to your GitHub repository settings:

```
DOCKER_USERNAME         - Docker Hub username
DOCKER_TOKEN            - Docker Hub access token
AWS_ACCESS_KEY_ID       - AWS IAM access key
AWS_SECRET_ACCESS_KEY   - AWS IAM secret key
EC2_SSH_PRIVATE_KEY     - EC2 .pem file contents
```

## Project Structure

```
Rag-fullstack-docker-AWS/
├── .github/
│   └── workflows/
│       └── ci-cd.yml              # GitHub Actions CI/CD pipeline
├── backend/
│   ├── main.py                    # FastAPI application
│   ├── models/                    # LLM integration modules
│   └── routes/                    # API endpoints
├── frontend/
│   ├── app.py                     # Streamlit application
│   └── pages/                     # Multi-page UI
├── tests/
│   ├── test_api.py
│   ├── test_rag.py
│   └── test_integration.py
├── Dockerfile                     # Multi-stage Docker build
├── docker-compose.yml             # Local development setup
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## API Endpoints

### Health Check
```bash
GET /health
```

### RAG Query
```bash
POST /query
Content-Type: application/json

{
  "question": "What is RAG?",
  "use_web_search": true,
  "model": "groq"
}
```

### Document Upload
```bash
POST /documents/upload
Content-Type: multipart/form-data

file: <pdf or txt file>
```

### List Documents
```bash
GET /documents
```

## Deployment

### AWS EC2 Setup

1. **Create EC2 Instance**
   ```bash
   aws ec2 run-instances \
     --image-id ami-0c55b159cbfafe1f0 \
     --instance-type t3.micro \
     --key-name your-key-pair
   ```

2. **Install Docker**
   ```bash
   ssh -i your-key.pem ec2-user@<instance-ip>
   
   # On EC2
   sudo yum update -y
   sudo yum install docker -y
   sudo usermod -aG docker ec2-user
   ```

3. **Configure GitHub Secrets**
   - Add AWS credentials to GitHub
   - Add EC2 .pem file content as `EC2_SSH_PRIVATE_KEY`
   - Add Docker credentials

4. **Push to main**
   - CI/CD pipeline automatically deploys to your EC2 instance

### Accessing Deployed Application

After deployment, access:
- **API**: `http://<ec2-ip>:8000`
- **Docs**: `http://<ec2-ip>:8000/docs`
- **Frontend**: `http://<ec2-ip>:8501`

## Monitoring

Check logs on EC2:

```bash
ssh -i your-key.pem ec2-user@<instance-ip>

# View container status
docker ps

# View logs
docker logs rag-fullstack -f

# View resource usage
docker stats rag-fullstack
```

## Troubleshooting

### Docker Image Push Fails
- Verify Docker Hub credentials in GitHub Secrets
- Check Docker Hub image name format

### EC2 Deployment Fails
- Verify EC2 instance is running: `aws ec2 describe-instances --instance-ids i-xxx`
- Check .pem file permissions: `chmod 600 ~/.ssh/aws-key.pem`
- Verify security group allows SSH (port 22) and HTTP (ports 8000, 8501)

### API Not Responding
- SSH to EC2 and check container: `docker logs rag-fullstack`
- Verify ports are mapped: `docker port rag-fullstack`
- Check EC2 security group inbound rules

## Development

### Running Tests Locally

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest tests/ -v --cov

# Run specific test
pytest tests/test_api.py -v
```

### Building Docker Image Locally

```bash
# Build image
docker build -t rag-fullstack:latest .

# Run container
docker run -p 8000:8000 -p 8501:8501 rag-fullstack:latest

# Push to Docker Hub
docker tag rag-fullstack:latest amith98480/rag-fullstack-docker-aws:latest
docker push amith98480/rag-fullstack-docker-aws:latest
```

## License

MIT License - see LICENSE file for details

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Support

For issues and questions:
- GitHub Issues: https://github.com/Amith-Ganta/Rag-fullstack-docker-AWS/issues
- Email: gantaamith007@gmail.com

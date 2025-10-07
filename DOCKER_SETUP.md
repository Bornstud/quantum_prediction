# 🐳 Docker Setup Guide for QuantumBCI

## Quick Start

### Option 1: Docker Compose (Recommended for Development)

1. **Build and run the entire stack:**
   ```bash
   docker-compose up --build
   ```

2. **Access the application:**
   - Open browser: `http://localhost:5000`

3. **Stop the services:**
   ```bash
   docker-compose down
   ```

4. **Clean up (remove volumes):**
   ```bash
   docker-compose down -v
   ```

---

### Option 2: Standalone Docker (Production)

1. **Build the image:**
   ```bash
   docker build -t quantumbci:latest .
   ```

2. **Run with external PostgreSQL:**
   ```bash
   docker run -p 5000:5000 \
     -e DATABASE_URL="postgresql://user:password@host:5432/dbname" \
     quantumbci:latest
   ```

3. **Run with environment file:**
   ```bash
   docker run -p 5000:5000 --env-file .env quantumbci:latest
   ```

---

## Environment Variables

Create a `.env` file in the project root:

```env
# PostgreSQL Configuration
POSTGRES_USER=admin
POSTGRES_PASSWORD=your_secure_password
POSTGRES_DB=quantumbci

# Database URL (auto-generated in docker-compose)
DATABASE_URL=postgresql://admin:password@postgres:5432/quantumbci
```

---

## Development Mode

To enable live code reloading during development:

1. Uncomment the volume mount in `docker-compose.yml`:
   ```yaml
   volumes:
     - .:/app  # Already uncommented
   ```

2. Changes to Python files will trigger Streamlit auto-reload

---

## Production Deployment

### Build Optimized Image:
```bash
docker build --no-cache -t quantumbci:prod .
```

### Push to Registry:
```bash
docker tag quantumbci:prod your-registry/quantumbci:latest
docker push your-registry/quantumbci:latest
```

### Deploy to Cloud:
```bash
# Example: Google Cloud Run
gcloud run deploy quantumbci \
  --image your-registry/quantumbci:latest \
  --port 5000 \
  --set-env-vars DATABASE_URL=$DATABASE_URL
```

---

## Troubleshooting

### Database Connection Issues:
- Ensure PostgreSQL is running: `docker-compose ps`
- Check DATABASE_URL is correctly set
- Verify network connectivity: `docker network ls`

### Port Already in Use:
```bash
# Change port in docker-compose.yml
ports:
  - "8080:5000"  # Use 8080 instead
```

### View Logs:
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f streamlit
docker-compose logs -f postgres
```

---

## Image Details

- **Base Image:** `python:3.11-slim-bookworm`
- **Exposed Port:** 5000
- **Health Check:** Enabled (30s interval)
- **User:** Non-root (streamlit:1000)
- **Estimated Size:** ~2-3GB (includes quantum ML libraries)

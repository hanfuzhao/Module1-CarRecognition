# Deployment Guide - Car Type Recognition

This guide covers deploying the application to cloud platforms.

## Quick Links

- **GitHub Repo:** https://github.com/hanfuzhao/Module1-CarRecognition
- **Recommended Deployment:** Render.com (free, easy, Python-native)

---

## Option 1: Deploy to Render.com (Recommended)

### Prerequisites
- GitHub account with repo access
- Render.com account (free)

### Steps

1. **Sign up / Log in to Render.com**
   - Visit https://render.com
   - Sign in with GitHub

2. **Create New Web Service**
   - Click "New +" → "Web Service"
   - Connect your GitHub account
   - Select `hanfuzhao/Module1-CarRecognition` repo
   - Branch: `main`

3. **Configure Service**
   - **Name:** `car-recognition` (or any name)
   - **Runtime:** Python 3.11
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn -w 2 -b 0.0.0.0:$PORT main:app`

4. **Environment Variables**
   - Click "Advanced"
   - Add: `PORT` = `10000` (Render assigns this)
   - Add: `FLASK_ENV` = `production`

5. **Plan**
   - Select "Free" plan
   - Deploy!

6. **Access Your App**
   - Render will provide a URL like: `https://car-recognition.onrender.com`
   - Open in browser → app is live!

### Notes
- Free tier has 15-min inactivity auto-stop (acceptable for assignment demo)
- App will keep running for ≥7 days after submission
- Models will be loaded if available in `/models` directory

---

## Option 2: Deploy to Heroku (Alternative)

### Prerequisites
- Heroku account (paid or free credits)
- Heroku CLI installed: `brew install heroku`

### Steps

```bash
# Login to Heroku
heroku login

# Create app
heroku create car-recognition-540

# Set environment
heroku config:set FLASK_ENV=production

# Deploy
git push heroku main

# View logs
heroku logs --tail

# Open app
heroku open
```

### Notes
- Free dyos were discontinued (Nov 2022)
- Requires Heroku credits or paid account

---

## Option 3: Docker + Any Platform (Manual)

### Build Docker Image
```bash
docker build -t car-recognition:latest .
```

### Test Locally
```bash
docker run -p 5000:5000 car-recognition:latest
# Access: http://localhost:5000
```

### Deploy to Docker Hub
```bash
# Tag
docker tag car-recognition:latest your-username/car-recognition:latest

# Push
docker push your-username/car-recognition:latest
```

### Deploy to Platforms Supporting Docker
- **Railway.app** (simple, ~$5/month)
- **DigitalOcean App Platform** (reliable)
- **AWS ECS/Fargate** (enterprise)
- **Google Cloud Run** (serverless, pay-per-use)

---

## Option 4: Local Deployment (For Testing)

### Using Flask Development Server
```bash
python main.py
# Access: http://localhost:5000
```

### Using Docker Compose
```bash
docker-compose up
# Access: http://localhost:5000
```

### Using Gunicorn (Production-like)
```bash
pip install gunicorn
gunicorn -w 2 -b 0.0.0.0:5000 main:app
```

---

## Post-Deployment Checklist

- [ ] App is accessible at public URL
- [ ] Can upload image and get prediction
- [ ] Top-5 predictions display correctly
- [ ] Confidence feedback works ("I'm not sure" message appears)
- [ ] No errors in logs
- [ ] App stays live for ≥7 days

---

## Troubleshooting

### Models Not Found
- **Issue:** "Could not load deep_learning model"
- **Solution:** Run `python setup.py` locally to train, then push models to repo (if <100MB total)
- **Alternative:** Use classical/naive models (no large files)

### Port Issues
- **Issue:** `Address already in use`
- **Solution:** Change port in config: `flask run --port 8000`

### Dependency Conflicts
- **Issue:** Torch installation fails
- **Solution:** Use `requirements-slim.txt` with CPU-only torch

### Timeout on Deploy
- **Issue:** Torch compilation takes >15 min
- **Solution:** Use pre-built wheels; Render may need "Standard" plan (paid)

---

## Monitoring & Logs

### Render.com
- Dashboard → Service → "Logs" tab
- Check for any errors during inference

### Heroku
```bash
heroku logs --tail
```

### Docker
```bash
docker logs <container-id>
```

---

## Scaling & Performance

### Current Setup
- 2 worker processes (Gunicorn)
- Suitable for ~10-20 concurrent users
- Inference time: ~50-200ms per image

### For Higher Load
- Increase workers: `gunicorn -w 4 ...`
- Use load balancer (platform-specific)
- Cache model in memory (already done)

---

## Security Notes

1. **Uploaded Images:** Stored in `static/uploads/` (temporary)
   - Consider adding cleanup job for old uploads
   - Or: Use ephemeral storage (no persistence)

2. **API Rate Limiting:** Not implemented
   - Add for production: `Flask-Limiter`

3. **CORS:** Disabled (localhost only for submission)
   - Enable if serving external clients

---

## Estimated Timeline

| Platform | Setup Time | Deploy Time | Cost |
|----------|-----------|------------|------|
| Render | 5 min | 2 min | Free |
| Heroku | 10 min | 5 min | $7+/month |
| Docker + Railway | 15 min | 3 min | ~$5/month |
| Local (testing) | 1 min | Instant | $0 |

---

## Getting Help

- **Render Docs:** https://render.com/docs
- **Flask Deployment:** https://flask.palletsprojects.com/deploy/
- **Docker Guide:** https://docs.docker.com/

---

## Summary

**For the 540 assignment:**

1. Deploy to Render.com (simplest)
2. Verify app is live: https://car-recognition.onrender.com
3. Share URL with instructor
4. App auto-stays live ≥7 days after submission
5. Grader can test live demo

**GitHub repo already has all code + deployment configs. Easy win!** 🚀

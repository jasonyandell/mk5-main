# Web Endpoints and Deployment

## Basic Web Endpoint

```python
@app.function(gpu="A10G", image=image)
@modal.web_endpoint(method="POST")
def predict(request: dict) -> dict:
    result = model.predict(request["input"])
    return {"prediction": result}
```

Returns URL like: `https://your-workspace--your-app--predict.modal.run`

## HTTP Methods

```python
@modal.web_endpoint(method="GET")
def health() -> dict:
    return {"status": "ok"}

@modal.web_endpoint(method="POST")
def predict(request: dict) -> dict:
    return {"result": process(request)}

@modal.web_endpoint(method="PUT")
def update(id: str, data: dict) -> dict:
    return {"updated": id}
```

## FastAPI Integration

Full FastAPI app with multiple routes:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

web_app = FastAPI()

class PredictRequest(BaseModel):
    input: list[float]
    model_version: str = "v1"

class PredictResponse(BaseModel):
    prediction: list[float]
    latency_ms: float

@web_app.get("/health")
def health():
    return {"status": "ok", "gpu": torch.cuda.is_available()}

@web_app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    start = time.time()
    result = model(request.input)
    latency = (time.time() - start) * 1000
    return PredictResponse(prediction=result, latency_ms=latency)

@web_app.get("/models")
def list_models():
    return {"models": ["v1", "v2"]}

@app.function(gpu="A10G", image=image)
@modal.asgi_app()
def fastapi_app():
    return web_app
```

## Classes for Inference Services

Stateful class with model loaded once:

```python
@app.cls(gpu="A10G", image=image, container_idle_timeout=300)
class InferenceService:
    @modal.enter()
    def load_model(self):
        import torch
        self.model = torch.load("/data/model.pt")
        self.model.eval()
        self.device = torch.device("cuda")
        self.model.to(self.device)

    @modal.web_endpoint(method="POST")
    def predict(self, request: dict) -> dict:
        with torch.no_grad():
            x = torch.tensor(request["input"]).to(self.device)
            result = self.model(x)
        return {"prediction": result.cpu().tolist()}

    @modal.web_endpoint(method="GET")
    def health(self) -> dict:
        return {"status": "ok", "model_loaded": self.model is not None}
```

## Development vs Production

### Development (hot-reload)

```bash
modal serve app.py
```

- Auto-reloads on code changes
- Shows logs in terminal
- Ephemeral deployment

### Production

```bash
modal deploy app.py
```

- Persistent deployment
- Survives terminal close
- Version tracked

### Check deployed apps

```bash
modal app list
# NAME          STATE      CREATED
# my-app        deployed   2025-01-10 12:00:00
```

## Scaling Configuration

```python
@app.function(
    gpu="A10G",
    # Scaling
    min_containers=0,       # Scale to zero when idle
    max_containers=10,      # Cap at 10 GPUs
    buffer_containers=1,    # Keep 1 warm for low latency

    # Container lifecycle
    container_idle_timeout=300,  # Shutdown after 5min idle
    timeout=60,             # Request timeout
)
@modal.web_endpoint()
def predict(request: dict):
    ...
```

## Keep Warm for Low Latency

```python
@app.function(
    gpu="A10G",
    buffer_containers=2,        # Keep 2 containers warm
    container_idle_timeout=600, # Keep alive 10min after last request
)
@modal.web_endpoint()
def low_latency_predict(request: dict):
    ...
```

Cold start times:
- Simple image: 1-5 seconds
- Large model loading: 30-120 seconds

Use `buffer_containers` to avoid cold starts for critical endpoints.

## Authentication

### API Key in Header

```python
from fastapi import Header, HTTPException

@web_app.post("/predict")
def predict(request: dict, x_api_key: str = Header(...)):
    if x_api_key != os.environ["API_KEY"]:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return {"result": model(request["input"])}

@app.function(secrets=[modal.Secret.from_name("api-keys")])
@modal.asgi_app()
def authenticated_app():
    return web_app
```

### Bearer Token

```python
from fastapi import Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

@web_app.post("/predict")
def predict(
    request: dict,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    if not verify_token(credentials.credentials):
        raise HTTPException(status_code=401)
    return process(request)
```

## CORS Configuration

```python
from fastapi.middleware.cors import CORSMiddleware

web_app = FastAPI()
web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://myapp.com"],
    allow_methods=["POST"],
    allow_headers=["*"],
)
```

## Streaming Responses

```python
from fastapi.responses import StreamingResponse

@web_app.post("/generate")
async def generate_stream(request: dict):
    async def stream():
        for token in model.generate_stream(request["prompt"]):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")
```

## File Uploads

```python
from fastapi import UploadFile, File

@web_app.post("/process-image")
async def process_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    result = model.predict(image)
    return {"prediction": result}
```

## Monitoring Deployed Endpoints

```bash
# Stream logs
modal app logs my-app

# List containers
modal container list

# Check GPU utilization
modal container exec <container-id> -- nvidia-smi
```

## Cost Optimization

1. **Scale to zero**: Set `min_containers=0` for low-traffic endpoints
2. **Right-size GPU**: Use L4/A10 for inference, not H100
3. **Batch requests**: Combine multiple inputs per request
4. **Container reuse**: Set reasonable `container_idle_timeout`
5. **Cache model weights**: Use volumes, not re-download each start

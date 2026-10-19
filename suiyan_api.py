# -*- coding: utf-8 -*-
"""SuiYan AI - FastAPI service exposing the RAG makeup-plan chain.

Endpoints:
  GET  /health                -> service status + store sizes
  POST /skin-profile/plan     -> SkinProfile JSON in, MakeupPlan JSON out
  GET  /products              -> full product catalog (for frontend cards)

Run from RAGProject root:
  py -3.12 -m uvicorn suiyan_api:app --host 0.0.0.0 --port 8000

Vision module / frontend contract:
  POST /skin-profile/plan  body:
    {
      "skin_type": "干性",            # 干性/油性/混合性/中性/敏感肌
      "skin_tone": "自然偏白（黄二白）",
      "undertone": "暖调",            # 冷调/暖调/中性调
      "concerns": ["法令纹", "眼周细纹"],
      "occasion": "日常通勤"
    }
"""
import base64
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI

from suiyan_chain import build_plan_chain, plan_to_json
from suiyan_product_store import SuiyanKnowledgeStore, SuiyanProductStore, load_products
from suiyan_schemas import SkinProfile
from suiyan_vision_utils import analyze_portrait_bytes

# OpenAI-compatible endpoint of DashScope, used for qwen3-asr-flash
_ASR_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# Globals initialized in lifespan (ingestion is idempotent, so restart is safe)
product_store: SuiyanProductStore = None  # type: ignore
knowledge_store: SuiyanKnowledgeStore = None  # type: ignore
run_plan = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Init vector stores + chain once at startup (ingest is idempotent)."""
    global product_store, knowledge_store, run_plan
    product_store = SuiyanProductStore()
    n_products = product_store.ingest()
    knowledge_store = SuiyanKnowledgeStore()
    n_knowledge = knowledge_store.ingest()
    run_plan, _ = build_plan_chain(product_store, knowledge_store)
    print(f"[suiyan_api] ready: products={n_products}, knowledge={n_knowledge}")
    yield
    # no explicit teardown needed (Chroma persists on disk)


app = FastAPI(
    title="SuiYan AI API",
    description="岁颜AI - 熟龄肌妆容方案生成服务 (欧莱雅美妆科技黑客松 赛题三)",
    version="0.1.0",
    lifespan=lifespan,
)

# Allow local frontend dev servers (Vite/React/静态页) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "suiyan-ai",
        "products": product_store.store._collection.count(),
        "knowledge_chunks": knowledge_store.store._collection.count(),
    }


@app.post("/skin-profile/plan")
def skin_profile_plan(profile: SkinProfile):
    """Full pipeline: SkinProfile -> filtered retrieval -> LLM plan
    -> anti-hallucination validation -> MakeupPlan JSON."""
    try:
        plan = run_plan(profile)
    except Exception as e:  # surface LLM/network errors as 502
        raise HTTPException(status_code=502, detail=f"plan generation failed: {e}")
    return JSONResponse(content=plan.model_dump(), media_type="application/json; charset=utf-8")


@app.get("/products")
def products():
    """Product catalog for frontend product cards."""
    return JSONResponse(content={"products": load_products()},
                        media_type="application/json; charset=utf-8")


@app.post("/vision/analyze")
def vision_analyze(file: UploadFile = File(...)):
    """Quantitative vision layer: portrait in -> ITA degree, undertone
    ratio, wrinkle density + suggested SkinProfile fields out.

    The frontend then lets the user pick skin_type (not quantifiable from a
    single photo) and POSTs the assembled SkinProfile to /skin-profile/plan.

    Sync `def`: OpenCV processing is CPU-bound/blocking; thread pool keeps
    the event loop free.
    """
    try:
        image_bytes = file.file.read()
        return analyze_portrait_bytes(image_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"vision analysis failed: {e}")


@app.post("/voice/transcribe")
def voice_transcribe(file: UploadFile = File(...)):
    """Voice request input: audio from MediaRecorder in -> transcript text
    out, via DashScope qwen3-asr-flash (OpenAI-compatible input_audio).

    NOTE: sync `def` (not async) on purpose — the OpenAI call below is
    blocking; FastAPI runs sync endpoints in the thread pool so the event
    loop (and /vision/analyze) stays responsive during ASR.
    """
    audio_bytes = file.file.read()
    if len(audio_bytes) < 1000:
        raise HTTPException(status_code=400, detail="audio too short, please speak longer")
    content_type = (file.content_type or "audio/webm").split(";")[0].strip()
    fmt = content_type.split("/")[-1]  # wav / webm / mp4 / ogg ...
    if fmt not in {"wav", "webm", "mp3", "mp4", "m4a", "ogg", "aac", "flac", "opus"}:
        fmt = "webm"
    b64 = base64.b64encode(audio_bytes).decode()
    data_uri = f"data:{fmt};base64,{b64}"
    try:
        client = OpenAI(
            api_key=os.environ.get("DASHSCOPE_API_KEY"),
            base_url=_ASR_BASE_URL,
        )
        r = client.chat.completions.create(
            model="qwen3-asr-flash",
            messages=[{
                "role": "user",
                "content": [{
                    "type": "input_audio",
                    "input_audio": {"data": data_uri, "format": fmt},
                }],
            }],
            extra_body={"asr_options": {"language": "zh"}},
            timeout=60,
        )
        text = (r.choices[0].message.content or "").strip()
        return {"text": text}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"transcribe failed: {e}")


# Static frontend (must be mounted AFTER API routes; index.html at "/")
class NoCacheStaticFiles(StaticFiles):
    """Serve index.html with no-store so frontend edits take effect on a
    normal refresh (browser otherwise reuses the cached old JS)."""
    def file_response(self, *args, **kwargs):
        resp = super().file_response(*args, **kwargs)
        resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        return resp


STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if os.path.isdir(STATIC_DIR):
    app.mount("/", NoCacheStaticFiles(directory=STATIC_DIR, html=True), name="static")

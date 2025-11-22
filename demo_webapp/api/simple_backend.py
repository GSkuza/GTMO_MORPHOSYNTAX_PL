#!/usr/bin/env python3
"""
SUPER PROSTY BACKEND Z CORS - TEST
"""
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="GTMØ Simple Backend")

# CORS - maksymalnie permissive
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return JSONResponse(
        content={"status": "ok", "message": "Simple backend with CORS"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "*",
        }
    )

@app.get("/health")
def health():
    return JSONResponse(
        content={"status": "healthy", "cors": "enabled"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "*",
        }
    )

@app.post("/test")
async def test_upload(file: UploadFile = File(...)):
    content = await file.read()
    return JSONResponse(
        content={
            "success": True,
            "filename": file.filename,
            "size": len(content),
            "text": content.decode('utf-8')[:100]
        },
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "*",
        }
    )

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("🧪 SIMPLE BACKEND WITH CORS")
    print("=" * 60)
    print("URL: http://127.0.0.1:8000")
    print("CORS: ENABLED for all origins")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000)

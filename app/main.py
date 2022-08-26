import time

import uvicorn
from pydantic import BaseModel
import fastapi
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from loguru import logger

from fastapi.responses import JSONResponse

from app.inference import Mouse_Detection


app = FastAPI(title="Mouse Video Detection API", description="This is API for project Mouse Video")


@app.get("/")
async def root():
    return {"message": "Mouse Video Detection"}


@app.post("/detect")
async def detect():
    pass


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=5000)

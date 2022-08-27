import time
import os

import aiofiles
import asyncio
import uvicorn
import fastapi

from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from loguru import logger
from tempfile import NamedTemporaryFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from inference import Mouse_Detection

app = FastAPI(title="Mouse Video Detection API", description="This is API for project Mouse Video")


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    mouse = Mouse_Detection()

    try:
        async with aiofiles.tempfile.NamedTemporaryFile("wb", delete=False) as temp:
            try:
                contents = await file.read()
                await temp.write(contents)
            except Exception:
                return {"message": "There was an error uploading the file"}
            finally:
                await file.close()

        list_p, detect_time = await run_in_threadpool(mouse.inference, temp.name)  # Pass temp.name to VideoCapture()
    except Exception:
        return {"message": "There was an error processing the file"}
    finally:
        os.remove(temp.name)

    return {
        "file_name": file.filename,
        "list port": list_p,
        "time": detect_time, }


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=5000)

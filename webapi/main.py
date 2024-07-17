import io
from typing import Dict

import uvicorn
from PIL import Image
from fastapi import FastAPI, UploadFile, File, APIRouter, Response

import config
import util
from model_adapter import ModelAdapter

router = APIRouter(
    prefix="/api/v1",
    tags=["v1"]
)

model_configs = config.parse_config()
adapter = ModelAdapter()


@router.get("/")
async def read_root():
    return Response(content="Hello, this is the root page of the API", status_code=200)


@router.get("/models")
async def get_models():
    model_list = [{"model_name": model["model_name"], "task": model["task"]} for model in model_configs]
    return model_list


def find_model_by_name(model_name: str) -> Dict | None:
    # 使用next()函数，如果没有找到匹配项，则返回None
    return next((model for model in model_configs if model["model_name"] == model_name), None)


@router.put("/restore")
async def restore(model_name: str, file: UploadFile = File(...), return_img_format: str = "PNG"):
    model_config = find_model_by_name(model_name)
    if model_config is None:
        return Response(content=f"Model {model_name} not found", status_code=400)
    if return_img_format not in ["PNG", "JPEG", "WEBP"]:
        return Response(content=f"Invalid image format: {return_img_format}", status_code=400)

    contents = await file.read()
    adapter.load_model(model_config)

    try:
        img_in = Image.open(io.BytesIO(contents))
        # img.verify()  # 验证图像是否有效

        img_tensor = util.pil2tensor(img_in)
        out_tensor, dur_time = adapter.run_forward(img_tensor)
        img_out = util.tensor2pil(out_tensor)

        img_byte_array = io.BytesIO()
        img_out.save(img_byte_array, format=return_img_format)
        img_byte_array = img_byte_array.getvalue()

    except Exception as e:
        print(e)
        return Response(content=f"error: {e}", status_code=500)

    # 返回图像
    return Response(content=img_byte_array, media_type=f"image/{return_img_format.lower()}", status_code=200)


app = FastAPI()
app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")

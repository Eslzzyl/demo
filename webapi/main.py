import io
from typing import Dict

import uvicorn
from PIL import Image
# 我们使用 fastapi 作为 Web 框架
from fastapi import FastAPI, UploadFile, File, APIRouter, Response

import config
import util
from model_adapter import ModelAdapter

# 定义一个共享的路由，这个路由具有固定的前缀 /api/v1
router = APIRouter(
    prefix="/api/v1",
    tags=["v1"]
)

# 解析 config 目录下放置的模型配置文件
model_configs = config.parse_config()
# 注册 model adapter
adapter = ModelAdapter()


@router.get("/")
async def read_root():
    """
    路径为 / 的 endpoint

    请求方法 GET

    请求参数 无

    返回值 固定的字符串 Hello, this is the root page of the API
    """
    return Response(content="Hello, this is the root page of the API", status_code=200)


@router.get("/models")
async def get_models():
    """
    路径为 /models 的 endpoint

    请求方法 GET

    请求参数 无
    
    返回值 所有已加载的模型的配置信息列表，形成一个 JSON 数组
    """
    model_list = [{"model_name": model["model_name"], "task": model["task"]} for model in model_configs]
    return model_list


def find_model_by_name(model_name: str) -> Dict | None:
    # 使用next()函数，如果没有找到匹配项，则返回None
    return next((model for model in model_configs if model["model_name"] == model_name), None)


@router.put("/restore")
async def restore(model_name: str, file: UploadFile = File(...), return_img_format: str = "PNG"):
    """
    路径为 /restore 的 endpoint

    请求方法 PUT

    请求参数 model_name: 模型的名称, file: 上传的图像, return_img_format: 返回的图像的格式
    
    返回值 处理后的图像，以 return_img_format 的格式编码
    """
    # 先通过模型名称获取对应模型的配置信息
    model_config = find_model_by_name(model_name)
    if model_config is None:
        return Response(content=f"Model {model_name} not found", status_code=400)
    if return_img_format not in ["PNG", "JPEG", "WEBP"]:
        return Response(content=f"Invalid image format: {return_img_format}", status_code=400)

    # 从 HTTP 请求中读取数据
    contents = await file.read()
    # 加载模型结构和权重
    adapter.load_model(model_config)

    try:
        # PIL 读取图像
        img_in = Image.open(io.BytesIO(contents))
        # img.verify()  # 验证图像是否有效

        # HWC 转 BCHW，[0, 255] 归一化到 [0, 1.0]
        img_tensor = util.pil2tensor(img_in)
        # 执行 forward 推理过程
        out_tensor, dur_time = adapter.run_forward(img_tensor)
        # BCHW 转 HWC，[0, 1.0] 反归一化到 [0, 255]
        img_out = util.tensor2pil(out_tensor)

        img_byte_array = io.BytesIO()
        # 将图像写入到字节流
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
    # 运行 WebAPI，监听来自本机内网的请求，端口 8000
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")

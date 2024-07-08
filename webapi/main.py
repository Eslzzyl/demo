import io
import time

import numpy as np
import torch
import uvicorn
from PIL import Image
from einops import rearrange
from fastapi import FastAPI, UploadFile, File, APIRouter, Response

from model.PReNet import PReNet

router = APIRouter(
    prefix="/api/v1",
    tags=["v1"]
)

use_GPU = torch.cuda.is_available()
print('use_GPU:', use_GPU)

model = PReNet()
checkpoint_path = 'pretrained/PReNet_Rain100L.pth'

if use_GPU:
    model = model.cuda()
    device = 'cuda'
else:
    device = 'cpu'
model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device(device)))
model.eval()


def infer(in_img):
    y = np.array(in_img).astype(np.float32) / 255.0
    y = rearrange(y, 'h w c -> 1 c h w')
    y = torch.Tensor(y)

    if use_GPU:
        y = y.cuda()

    with torch.no_grad():
        if use_GPU:
            torch.cuda.synchronize()
        start_time = time.time()

        out, _ = model(y)
        out = torch.clamp(out, 0., 1.)

        if use_GPU:
            torch.cuda.synchronize()
        end_time = time.time()
        dur_time = end_time - start_time

        if use_GPU:
            save_out = np.uint8(255 * out.data.cpu().numpy())  # back to cpu
        else:
            save_out = np.uint8(255 * out.data.numpy())

        save_out = rearrange(save_out, '1 c h w -> h w c')
    return save_out


@router.get("/")
async def read_root():
    return {"msg": "Hello World"}


@router.post("/restore")
async def create_upload_file(file: UploadFile = File(...)):
    contents = await file.read()

    try:
        img = Image.open(io.BytesIO(contents))
        # img.verify()  # 验证图像是否有效

        out = infer(img)

        # 将numpy数组转换回图像
        img = Image.fromarray(out)
        img_byte_array = io.BytesIO()
        img.save(img_byte_array, format='JPEG')
        img_byte_array = img_byte_array.getvalue()

    except Exception as e:
        print(e)
        return {"error": str(e)}

    # 返回图像
    return Response(content=img_byte_array, media_type="image/jpg")


app = FastAPI()
app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="debug")

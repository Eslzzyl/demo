from io import BytesIO

import gradio as gr
import requests
from PIL import Image, ImageDraw

base_url = "http://127.0.0.1:8000"


def request(image: Image, model) -> Image:
    # 将PIL.Image转换为字节流
    image_byte_array = BytesIO()
    image.save(image_byte_array, format='JPEG')
    image_byte_array = image_byte_array.getvalue()

    # 设置文件数据
    files = {'file': ('low.jpg', image_byte_array, 'image/jpeg')}

    try:
        response = requests.post(f"{base_url}/api/v1/restore", files=files)
        if response.status_code == 200:
            img_data = response.content
            img = Image.open(BytesIO(img_data))
            return img
        else:
            raise gr.Error(f"请求失败，状态码：{response.status_code}")
    except Exception as e:
        raise gr.Error(str(e), duration=10)


model_list = ["PReNet-Rain100L"]

gr_input = [
    gr.Image(
        type='pil',
        label='输入图片'
    ),
    gr.Dropdown(
        choices=model_list,
        value="PReNet-Rain100L",
        multiselect=False,
        label="选择复原模型",
    )
]

gr_output = [
    gr.Image(
        type='pil',
        label='复原效果图'
    ),
]

demo = gr.Interface(
    fn=request,
    inputs=gr_input,
    outputs=gr_output,
    submit_btn='提交',
    clear_btn='清除',
    allow_flagging='never',
    title='图像复原演示',
).queue(default_concurrency_limit=2)

demo.queue(default_concurrency_limit=4).launch(max_file_size='50mb')

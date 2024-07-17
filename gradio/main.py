from io import BytesIO
import os
import gradio as gr
import requests
from PIL import Image

base_url = "http://127.0.0.1:8000/api/v1"


def get_model_list(task):
    global models
    if task == "default":
        m_list = [model['model_name'] for model in models]
        return gr.Dropdown(choices=m_list, value=m_list[0])
    else:
        m_list = [model['model_name'] for model in models if model['task'] == task]
        return gr.Dropdown(choices=m_list, value=m_list[0])


def request(image: Image, model_name: str) -> Image:
    # 将PIL.Image转换为字节流
    image_byte_array = BytesIO()
    image.save(image_byte_array, format='JPEG')
    image_byte_array = image_byte_array.getvalue()

    # 将 model_name 中的加号替换为 %2B
    model_name = model_name.replace("+", "%2B")

    # 设置文件数据
    files = {'file': ('low.jpg', image_byte_array, 'image/jpeg')}

    try:
        request_url = f"{base_url}/restore?model_name={model_name}&return_img_format=JPEG"
        response = requests.put(request_url, files=files)
        if response.status_code == 200:
            img_data = response.content
            img = Image.open(BytesIO(img_data))
            return img
        else:
            raise gr.Error(f"请求失败，状态码: {response.status_code}, 消息: {response.text}", duration=10)
    except Exception as e:
        raise gr.Error(str(e), duration=10)


models = requests.get(f"{base_url}/models").json()
if len(models) == 0:
    raise gr.Error("未获取到模型列表")
task_list = list(set([model['task'] for model in models]))
task_list.append("default")
model_list = [model['model_name'] for model in models]

example_list = []
for root, dirs, files in os.walk("samples"):
    for file in files:
        example_list.append(os.path.join(root, file))

with gr.Blocks(title='Gradio 演示') as demo:
    with gr.Tab("图像复原演示"):
        with gr.Column():
            with gr.Row():
                input_image = gr.Image(type='pil', label='输入图片', height="45%")
                output_image = gr.Image(type='pil', label='复原效果图', height="45%")
            example = gr.Examples(examples=example_list, label='示例图片', inputs=[input_image])
            with gr.Row():
                task_dropdown = gr.Dropdown(choices=task_list, value="default", multiselect=False, label="选择任务")
                model_dropdown = gr.Dropdown(choices=model_list, value=model_list[0], multiselect=False,
                                             label="选择复原模型")
            with gr.Row():
                submit_btn = gr.Button(value='提交', variant='primary')
                clear_btn = gr.Button(value='清除')

        task_dropdown.input(fn=get_model_list, inputs=[task_dropdown], outputs=[model_dropdown])
        submit_btn.click(fn=request, inputs=[input_image, model_dropdown], outputs=[output_image])

if __name__ == '__main__':
    demo.queue(default_concurrency_limit=4).launch(max_file_size='50mb')

from io import BytesIO
import os
import gradio as gr
import requests
from PIL import Image

# 定义一个 base url，这个 url 对于所有请求都是一致的，所以可以写成常量。
base_url = "http://127.0.0.1:8000/api/v1"


def on_task_dropdown_input(task):
    """
    当 task_dropdown（用于管理复原任务的下拉菜单）的值被用户改变时，调用此函数
    """
    global models, example_list
    # 根据具体的 task 来改变 model_dropdown（用于管理复原模型的下拉菜单）的可选值
    if task == "default":
        m_list = [model['model_name'] for model in models]
        return gr.Dropdown(choices=m_list, value=m_list[0])
    else:
        m_list = [model['model_name'] for model in models if model['task'] == task]
        return gr.Dropdown(choices=m_list, value=m_list[0])


def get_input_text(image_path):
    """
    当输入图像改变时，调用此函数
    """
    # 获取图像的分辨率并返回，用于填充 HTML 组件
    if image_path is not None:
        image = Image.open(image_path)
        return f"<p>分辨率: {image.width} x {image.height}</p>"
    else:
        return "<p>分辨率: </p>"


def request(image_path, model_name: str) -> Image:
    """
    向 WebAPI 发起图像复原请求
    """
    if image_path is None:
        # gr.Error 将会通过红色对话框的形式展示给用户
        raise gr.Error("未选择图片")
    image = Image.open(image_path)
    # 将PIL.Image转换为字节流
    image_byte_array = BytesIO()
    image.save(image_byte_array, format='JPEG')
    image_byte_array = image_byte_array.getvalue()

    # 将 model_name 中的加号替换为 %2B，这是为了处理某些模型名称中带有加号的情况
    # 当然，不建议在模型名称中使用加号。
    model_name = model_name.replace("+", "%2B")

    # 设置文件数据
    files = {'file': ('low.jpg', image_byte_array, 'image/jpeg')}

    try:
        # 形成请求 URL
        request_url = f"{base_url}/restore?model_name={model_name}&return_img_format=JPEG"
        # 通过 HTTP PUT 方法向指定 URL 发送请求
        response = requests.put(request_url, files=files)
        # 如果状态码是 200，则从响应中读取图像，然后显示
        if response.status_code == 200:
            img_data = response.content
            img = Image.open(BytesIO(img_data), formats=['JPEG'])
            # 这里偷了个懒，实际上的处理耗时应该是 WebAPI 计算的推理耗时，这里直接计算了从请求发出到收到响应的时间，还加上了额外的网络传输时间
            text = f"<p>分辨率: {img.width} x {img.height}</p><p>处理耗时: {response.elapsed.total_seconds()}s</p>"
            return img, text
        else:
            raise gr.Error(f"请求失败，状态码: {response.status_code}, 消息: {response.text}", duration=10)
    except Exception as e:
        raise gr.Error(str(e), duration=10)


# 在程序启动时就获取模型列表
models = requests.get(f"{base_url}/models").json()
if len(models) == 0:
    raise gr.Error("未获取到模型列表")
# 然后过滤出任务列表
task_list = list(set([model['task'] for model in models]))
task_list.append("default")
model_list = [model['model_name'] for model in models]

# 事先准备好的一些实例图像，供用户直接点击查看
example_list = []
for root, dirs, files in os.walk("samples"):
    for file in files:
        example_list.append(os.path.join(root, file))

with gr.Blocks(title='Gradio 演示') as demo:
    with gr.Tab("图像复原演示"):
        with gr.Column():
            with gr.Row():
                input_image = gr.Image(type='filepath', label='输入图片', height="45%")
                output_image = gr.Image(type='pil', label='复原效果图', height="45%", interactive=True, sources=[])
            with gr.Row():
                input_text = gr.HTML(value="<p>分辨率: </p>")
                output_text = gr.HTML(value="<p>分辨率: </p><p>处理耗时: </p>")
            example = gr.Examples(examples=example_list, label="示例图片", inputs=[input_image])
            with gr.Row():
                task_dropdown = gr.Dropdown(choices=task_list, value="default", multiselect=False, label="选择任务")
                model_dropdown = gr.Dropdown(choices=model_list, value=model_list[0], multiselect=False,
                                             label="选择复原模型")
            with gr.Row():
                submit_btn = gr.Button(value='提交', variant='primary')
                clear_btn = gr.ClearButton(value='清除', components=[input_image, output_image])

        task_dropdown.input(fn=on_task_dropdown_input, inputs=[task_dropdown], outputs=[model_dropdown])
        submit_btn.click(fn=request, inputs=[input_image, model_dropdown], outputs=[output_image, output_text])
        input_image.change(fn=get_input_text, inputs=[input_image], outputs=[input_text])

if __name__ == '__main__':
    # 启动 gradio 程序
    # 注意，在通过 Nginx 反向代理部署时，launch 方法需要添加额外的 root_path 参数。
    # 例如，我们希望通过 demo.honorvision.cn/gradio 访问这个 gradio 程序时，
    # 需要设置 xxx.launch(max_file_size='50mb', root_path='/gradio')
    # 详见 https://www.gradio.app/guides/running-gradio-on-your-web-server-with-nginx
    demo.queue(default_concurrency_limit=4).launch(max_file_size='50mb')

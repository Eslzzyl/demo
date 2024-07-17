# WebAPI 使用说明

## 模型接口

当前 WebAPI 仅支持图像到图像的模型，例如图像复原模型。仅支持 PyTorch 模型。

### 命名和文件组织

一个模型应当具有：
- **一个** `.json` 模型配置文件，文件名任意，其中包含了该模型可能执行的所有任务，放置在 `webapi/config/` 目录中。
- **一个** `.py` 模型代码文件，文件名和其中的模型主类的类名应当与配置文件中的 `class_name` 字段保持一致。放置在 `webapi/model/` 目录中。
- **至少一个**模型权重文件，放置在 `webapi/pretrained/` 目录中。可以自行建立子目录，但需要保证在模型配置文件中的 `pretrained_path` 字段给出正确的路径。
  - 一些通用图像复原模型（如 Restormer）可能支持多种复原任务，因此会具有多个模型权重文件。每个模型权重文件对应模型配置文件中的一个项。这些针对不同复原任务的模型对用户来说是不同的模型，但它们实际上共用同一套代码。


### `__init__`

模型的主类应当具有一个 `__init__` 方法，其中接收必要的参数，也可以不接收任何参数。如果接收了额外的参数，则需要在模型配置文件的 `init_params` 字段中配置这些参数。

### `forward`

新添加的模型应当在 `forward` 方法中接收一个通道顺序为 `[N, C, H, W]`，值区间为 `[0, 1.]` 的 `torch.Tensor` 作为输入，并返回一个遵循同样格式的 `torch.Tensor` 作为输出图像。在模型推理过程中，`N` 的值恒定为 1。`forward` 方法不应当有任何额外的输入和输出数据。

WebAPI 的模型适配器（见 `webapi/model_adapter.py`）目前会自动对模型的输出进行 `torch.clamp` 操作，确保其取值位于 `[0, 1.]` 区间。

一些旧式的模型可能会在 `[0, 255]` 区间对图像进行处理，此时应当在模型 `forward` 方法的开头将输入乘以 255，并在返回前将输出除以 255。

## 配置文件

目前，WebAPI 采用 JSON 配置文件，每个模型的配置文件都应当包含一个 JSON 数组，其中可以包含一个或多个 JSON 对象，每个 JSON 对象对应一个具体的任务。对于用户来说，这些任务可以算作是不同的模型。

配置文件目前支持的字段有：

| 字段                 | 值类型 | 是否必须 | 说明                                                                                                                                                                                                                                                       |
| -------------------- | ------ | -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `model_name`         | string | 是       | 模型的名称，供其他访问 WebAPI 的程序指定模型使用。在程序中注册的所有模型必须具有独一无二的 `model_name` 以便 WebAPI 进行区分。该字段不能包含 `+` 等需要在 URL 中进行转义的字符。                                                                           |
| `class_name`         | string | 是       | 模型的代码文件名和主类名（二者需要具备相同的名字）。在模型切换时，WebAPI 的模型适配器将会执行类似 `from model.[class_name] import class_name` 的 Python 代码来导入模型。                                                                                   |
| `pretrained_path`    | string | 是       | 模型的权重文件的位置。尽管此处对权重文件放置在何处并没有限制，但应当将所有权重文件都放置在 `webapi/pretrained` 目录中以便于管理。                                                                                                                          |
| `state_dict_mapping` | string | 否       | 某些模型保存的权重文件中不仅仅包含模型的参数，还包含了一些其他信息（如 epoch、optimizer 等），此时模型的参数会使用一个特定的前缀来存储，此处用于适配这种前缀。具体可参看模型的原始代码中权重的加载部分。如果缺少该字段，则默认权重文件中仅存储了模型参数。 |
| `init_params`        | object | 否       | 某些模型的 `__init__` 方法会具有额外的参数，例如适用于多种复原任务的模型往往对不同任务会有不同的参数。此处用于传入这种参数。可参考配置中的 `MPRNet.json` 或 `Restormer.json`。                                                                             |
| `resize_factor`      | number | 否       | 许多模型要求输入图像的长和宽是某个数字（如 8 或 16）的整数倍。此处用于设置这种数字。如果存在该字段，则模型适配器将对图像执行镜像扩展，并在通过模型后再裁剪掉扩展出的部分。                                                                                 |
| `task`               | string | 是       | 模型执行的任务。调用 WebAPI 的前端程序可以根据任务来分类呈现模型。                                                                                                                                                                                         |

WebAPI 程序启动时，将自动读取 `config` 目录下的所有 `.json` 文件，对于每个

## 代码文件

每个模型的结构应当编写进**单一**的 `.py` 文件，并放置在 `webapi/model/` 目录中。代码文件名和模型主类的类名应当保持一致，例如，编写一个 `PReNet.py` 文件，其中的主类是 `PReNet`。代码文件引入的任何外部依赖都必须合并到同一个文件。

## WebAPI 访问点

### GET `/`

- 请求参数：无
- 响应：固定的文本 `Hello, this is the root page of the API`

该访问点主要用于验证 WebAPI 是否正常工作。

### GET `/models`

- 请求参数：无
- 响应：一个 JSON 数组，其中包含了所有可用模型的 `model_name` 和 `task` 信息。

例：
```json
[
  {"model_name":"SwinIR-lightweightSR-x2","task":"super-resolution"},
  {"model_name":"MPRNet-deraining","task":"deraining"},
  {"model_name":"MPRNet-denosing","task":"denoising"},
  {"model_name":"Restormer-deraining","task":"deraining"},
  {"model_name":"PReNet-Rain100L","task":"deraining"},
  {"model_name":"Zero-DCE","task":"low-light-enhancement"},
  {"model_name":"MIMO-UNet","task":"motion-deblurring"},
  {"model_name":"RIDNet","task":"denoising"}
]
```

### PUT `/restore`

- 请求参数：
  - `model_name`：是 URL 中的参数，指出需要使用的模型名称。该字段与 WebAPI 配置文件中的 `model_name` 对应。
  - `return_img_format`：期望 WebAPI 返回的图像格式。目前支持的格式包括 JPEG、PNG 和 WebP。传入参数时应当使用全大写。如果不指定该参数，则默认返回 PNG 格式。为了节省网络开销，建议使用带压缩的 JPEG 格式。
  - `file`：需要处理的图像。应当在请求体中提交，请求体类型是 `multipart/formdata`。
- 响应：处理后的图像的字节表示。

使用 `requests` 库的请求示例：
```python
import requests

base_url = "http://127.0.0.1:8000/api/v1"
model_name = "PReNet-Rain100L"
try:
    request_url = f"{base_url}/restore?model_name={model_name}&return_img_format=JPEG"
    response = requests.put(request_url, files=files)
    if response.status_code == 200:
        img_data = response.content
        img = Image.open(BytesIO(img_data))
        // 略
    else:
        print(f"请求失败，状态码: {response.status_code}, 消息: {response.text}")
except Exception as e:
    print(e)
```
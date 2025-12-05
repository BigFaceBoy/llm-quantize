# 0、环境信息
- nvidia L40
- python 3.12
- llmcompressor 0.8.1
- transformers 4.56.2
- torch 2.8.0
- torchvision 0.23.0

# 一、量化
- 1、从 InternVL3_5-8B 下载`chat_template.jinja`， 并将其放到本地下载的 InternVL3-8B 的根目录下
- 2、将 InternVL3-8B 下 configuration_internvl_chat.py 中的所有 int() 转换改为 math.floor()
- 3、运行 fp8_gptq.py 

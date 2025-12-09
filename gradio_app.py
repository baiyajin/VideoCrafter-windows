import os
import sys
import warnings
import io

# 抑制所有警告
warnings.filterwarnings('ignore')
os.environ['XFORMERS_IGNORE_MISSING_TRITON'] = '1'

# 在导入可能使用 xformers 的模块之前，临时抑制 stderr 来捕获 triton 错误信息
_original_stderr = sys.stderr
_suppressed_stderr = io.StringIO()
sys.stderr = _suppressed_stderr

# 预导入 xformers 以触发警告（在抑制 stderr 的情况下）
try:
    import xformers
except:
    pass

# 恢复 stderr
sys.stderr = _original_stderr

# 添加 lvdm 到路径（在导入 Text2Video 之前，因为它会导入 lvdm 模块）
sys.path.insert(1, os.path.join(sys.path[0], 'lvdm'))

import gradio as gr
from scripts.gradio.t2v_test import Text2Video

t2v_examples = [
    ['an elephant is walking under the sea, 4K, high definition',50, 12,1, 16],
    ['an astronaut riding a horse in outer space',25,12,1,16],
    ['a monkey is playing a piano',25,12,1,16],
    ['A fire is burning on a candle',25,12,1,16],
    ['a horse is drinking in the river',25,12,1,16],
    ['Robot dancing in times square',25,12,1,16],                    
]


def videocrafter_demo(result_dir='./tmp/'):
    text2video = Text2Video(result_dir)
    with gr.Blocks(analytics_enabled=False) as videocrafter_iface:
        gr.Markdown("<div align='center'> <h2> VideoCrafter2: Overcoming Data Limitations for High-Quality Video Diffusion Models </span> </h2> \
                     <a style='font-size:18px;color: #000000' href='https://github.com/AILab-CVC/VideoCrafter'> Github </div>")
        
        #######t2v#######
        with gr.Tab(label="Text2Video"):
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        input_text = gr.Text(label='Prompts')
                        with gr.Row():
                            steps = gr.Slider(minimum=1, maximum=60, step=1, elem_id=f"steps", label="Sampling steps", value=50)
                            eta = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, label='ETA', value=1.0, elem_id="eta")
                        with gr.Row():
                            cfg_scale = gr.Slider(minimum=1.0, maximum=30.0, step=0.5, label='CFG Scale', value=12.0, elem_id="cfg_scale")
                            fps = gr.Slider(minimum=4, maximum=32, step=1, label='fps', value=16, elem_id="fps")
                        send_btn = gr.Button("Send")
                    with gr.Column():
                        output_video_1 = gr.Video(width=512)
                gr.Examples(examples=t2v_examples,
                            inputs=[input_text,steps,cfg_scale,eta],
                            outputs=[output_video_1],
                            fn=text2video.get_prompt,
                            cache_examples=False)
                        #cache_examples=os.getenv('SYSTEM') == 'spaces')
            send_btn.click(
                fn=text2video.get_prompt, 
                inputs=[input_text,steps,cfg_scale,eta,fps],
                outputs=[output_video_1],
            )

    return videocrafter_iface

if __name__ == "__main__":
    result_dir = os.path.join('./', 'results')
    videocrafter_iface = videocrafter_demo(result_dir)
    videocrafter_iface.queue(max_size=10, default_concurrency_limit=1)
    videocrafter_iface.launch()
    # videocrafter_iface.launch(server_name='0.0.0.0', server_port=80)
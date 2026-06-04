"""
debug_webcam.py — 웹캠 컴포넌트 단독 테스트.
실행: python demo/debug_webcam.py
브라우저 http://127.0.0.1:7861 접속 후 카메라 버튼 클릭.
"""
import gradio as gr
import numpy as np

print(f"Gradio version: {gr.__version__}")

def show_frame(frame):
    if frame is None:
        return "frame=None — 캡처 안 됨"
    return f"frame shape={frame.shape}, dtype={frame.dtype}, min={frame.min()}, max={frame.max()}"

with gr.Blocks() as demo:
    gr.Markdown("## 웹캠 디버그\n카메라 버튼 클릭 → 촬영 → 아래 텍스트 확인")
    cam = gr.Image(sources=["webcam"], type="numpy", label="웹캠")
    out = gr.Textbox(label="수신 프레임 정보")
    cam.change(show_frame, inputs=cam, outputs=out)

demo.launch(server_port=7862)

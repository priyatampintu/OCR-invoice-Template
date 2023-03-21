
import argparse

import gradio as gr
import torch
from PIL import Image
import os
from donut import DonutModel


def demo_process_vqa(input_img, question):
    global pretrained_model, task_prompt, task_name
    input_img = Image.fromarray(input_img)
    user_prompt = task_prompt.replace("{user_input}", question)
    output = pretrained_model.inference(input_img, prompt=user_prompt)["predictions"][0]
    return output


def demo_process(input_img):
    global pretrained_model, task_prompt, task_name
    input_img = Image.fromarray(input_img)
    output = pretrained_model.inference(image=input_img, prompt="<s_sroie-donut>")["predictions"][0]
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="cord-v2")
    parser.add_argument("--pretrained_path", type=str, default="result/train_sroie/20230320_103158")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--url", type=str, default=None)
    parser.add_argument("--sample_img_path", type=str)
    args, left_argv = parser.parse_known_args()

    task_name = args.task
    if "docvqa" == task_name:
        task_prompt = "<s_docvqa><s_question>{user_input}</s_question><s_answer>"
    else:  # rvlcdip, cord, ...
        task_prompt = f"<s_{task_name}>"

    example_sample = []
    if args.sample_img_path:
        example_sample.append(args.sample_img_path)

    pretrained_model = DonutModel.from_pretrained(args.pretrained_path)

    if torch.cuda.is_available():
        pretrained_model.half()
        device = torch.device("cuda")
        pretrained_model.to(device)

    pretrained_model.eval()

    demo = gr.Interface(
        fn=demo_process_vqa if task_name == "docvqa" else demo_process,
        inputs=["image", "text"] if task_name == "docvqa" else "image",
        outputs="json",
        title=f"OCR for Invoice Template",
        examples=[
            [os.path.join(os.path.dirname(__file__), "examples/example-1.jpg")],
            [os.path.join(os.path.dirname(__file__), "examples/example-2.jpg")],
            [os.path.join(os.path.dirname(__file__), "examples/example-3.jpg")],
            [os.path.join(os.path.dirname(__file__), "examples/example-4.jpg")],
            [os.path.join(os.path.dirname(__file__), "examples/example-5.jpg")],
    ],
    )
    demo.launch(debug=True, enable_queue=True, server_name="0.0.0.0", server_port=7861)

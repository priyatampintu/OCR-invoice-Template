# OCR for Invoice Template(Documentation)

This is the OCR Transformer model(Donut) for AskGalore Digital India Pvt. Ltd where will extract Invoice Number and Total Amount from different different invoice bill like restuarent bill, fuel bill, mall bill etc and at the end will deploy project on server using Docker and gradio.
I tried lots of library:
  1. Pytesseract
  2. PaddleOcr
  3. Yolo + paddleOcr
  4. Transformer

But out of all these library best results is getting from OCR transformer model for this type of use case because invoice has dynamic templates and has not fixed position or area. Transformers deep learning model architecture to label words or answer given questions based on an image of a document (for example, you might either highlight and label the account number by annotating the image itself.Libraries such as HuggingFace’s transformers make it easier to work with open-source transformers models. In other words, it encodes the image (split into patches using a Swin Transformer) into token vectors it can then decode, or translate, into an output sequence in the form of a data structure (which can then be further parsed into JSON) using the BART decoder model, publicly pretrained on multilingual datasets. Any prompts fed into the model at inference time can also be decoded as well in the same architecture.
![Logo](https://raw.githubusercontent.com/priyatampintu/OCR-invoice-Template/main/examples/archtitecture.jpg)

![Logo](https://raw.githubusercontent.com/priyatampintu/OCR-invoice-Template/main/examples/invoice_demo.jpg) 
## Try it Demo

http://216.48.191.9:7861/


## Tutorial

This tutorial was tested on Google Cloud Comute Engine and the VM has the following specifications:

```bash
16 vCPU
60gb Ram
1 x NVIDIA Tesla T4
ubuntu 20.0.4
python 3.9
torch>=1.7
cuda 11.0
```
## Setup Environment 
git clone https://github.com/priyatampintu/OCR-invoice-Template.git

Install ananconda environment
```bash
  cd OCR-invoice-Template
  conda create -n invoice_ocr python=3.7
  conda activate invoice_ocr
  pip install -r requirements.txt
```
## STEP 1. Finetuning Donut on a custom dataset

It contains 100 images  to demonstrate Donut’s effectiveness. It’s a smaller dataset than CORD (which contains ~1000 images), and also much fewer labels (only Invoice NUmber and Total Amount).

Images and lablel's name should be same with jpg and json format.

![Logo](https://raw.githubusercontent.com/priyatampintu/OCR-invoice-Template/main/examples/sample.jpg)

## STEP 2. Data Labeling

  > tree dataset_name
  dataset_name
  ├── test
  │   ├── metadata.jsonl
  │   ├── {image_path0}
  │   ├── {image_path1}
  │             .
  │             .
  ├── train
  │   ├── metadata.jsonl
  │   ├── {image_path0}
  │   ├── {image_path1}
  │             .
  │             .
  └── validation
      ├── metadata.jsonl
      ├── {image_path0}
      ├── {image_path1}
              .
              .

  > cat dataset_name/test/metadata.jsonl
  {"file_name": {image_path0}, "ground_truth": "{\"gt_parse\": {ground_truth_parse}, ... {other_metadata_not_used} ... }"}
  {"file_name": {image_path1}, "ground_truth": "{\"gt_parse\": {ground_truth_parse}, ... {other_metadata_not_used} ... }"}
     .
     .


## STEP 4. Modify the label_map.txt file

Enter the label and number.

```bash
      dataset_name
    ├── test
    │   ├── metadata.jsonl
    │   ├── {image_path0}
    │   ├── {image_path1}
    │             .
    │             .
    ├── train
    │   ├── metadata.jsonl
    │   ├── {image_path0}
    │   ├── {image_path1}
    │             .
    │             .
    └── validation
        ├── metadata.jsonl
        ├── {image_path0}
        ├── {image_path1}
                  .
                  .
```
Here’s the script I used to transform the data into JSON lines files, as well as copy the images into their respective folders:

```bash
      import os
      import json
      import shutil
      import random
      from tqdm.notebook import tqdm
      lines = []
      images = []
      q=1
      for ann in tqdm(random.sample(os.listdir("key"), 20)):
          #print(ann)
          if ann[:-4] + "jpg" not in os.listdir("sroie-donut/train") and ann[:-4] + "jpg" not in os.listdir("sroie-donut/validation"):
              print(q, ann)
              q+=1
              if ann != ".ipynb_checkpoints":
                  with open("key/" + ann) as f:
                      data = json.load(f)
              images.append(ann[:-4] + "jpg")
              line = {"gt_parse": data}
              lines.append(line)
              with open("./sroie-donut/test/metadata.jsonl", 'w') as f:
                  for i, gt_parse in enumerate(lines):
                      line = {"file_name": images[i], "ground_truth": json.dumps(gt_parse)}
                      f.write(json.dumps(line) + "\n")
              shutil.copyfile("img/" + images[i], "./sroie-donut/test/" + images[i])
```
I simply ran this script three times, changing the names of the folders and the list slice ([:100]) each time, so that I had 100 examples in train and 20 examples each in validation and test.

## STEP 5. Training OCR Transformer model

```bash
# import library

python train.py --config config/train_sroie.yaml
```

![Logo](https://raw.githubusercontent.com/priyatampintu/OCR-invoice-Template/main/examples/training_0cr.jpg)

After successfully trained your model. Weight file (pytorch_model.bin) saved in directory(result/train_sroie/20230320_103158/pytorch_model.bin).

## STEP 6. Model evaluation and performance
There are two major parameters to measure object detection model's perforamnce:

    1. mAP(Mean Average Precision)
    2. Performance Matrix(Accuracy, Precision, Recall)

![Logo](https://raw.githubusercontent.com/priyatampintu/image-clssification-shirtsandtshrts/master/examples/confusion_matrix.png)

![Logo](https://raw.githubusercontent.com/priyatampintu/image-clssification-shirtsandtshrts/master/examples/results.png)

![Logo](https://raw.githubusercontent.com/priyatampintu/image-clssification-shirtsandtshrts/master/examples/R_curve.png)

## STEP 7. Test model
Downlod weight file from https://drive.google.com/file/d/1Z7guW2Mxqim7FXPry4i2TroZ0L1w5CSh/view?usp=share_link    
    from donut import DonutModel
    from PIL import Image
    import torch
    model = DonutModel.from_pretrained("result/train_sroie/20230320_103158")
    if torch.cuda.is_available():
      model.half()
      device = torch.device("cuda")
      model.to(device)
    else:
      model.encoder.to(torch.bfloat16)
    model.eval()
    image = Image.open("examples/example-1.jpg").convert("RGB")
    output = model.inference(image=image, prompt="<s_sroie-donut>")
    print(output)

![Logo](https://raw.githubusercontent.com/priyatampintu/OCR-invoice-Template/main/examples/result1.jpg)
![Logo](https://raw.githubusercontent.com/priyatampintu/OCR-invoice-Template/main/examples/result2.jpg)
![Logo](https://raw.githubusercontent.com/priyatampintu/OCR-invoice-Template/main/examples/result3.jpg)
![Logo](https://raw.githubusercontent.com/priyatampintu/OCR-invoice-Template/main/examples/result4.jpg)

## STEP 8. Deploy Model on Server 

There are multiple ways to deploy project on server like restAPI:

    1. Flask
    2. Streamlit 
    3. Fastapi
    4. gradio

### using Gradio(CLI)

     # Please open port(7861) to run gradio API
     python app.py


### Deploy using Docker 

    1. docker build -t invoice_ocr .

    # run docker container with nvidia-gpu support
    2. docker run --gpus all -p 7861:7861 invoice_ocr
    
    # run docker container with cpu support
    3. docker run -p 7861:7861 invoice_ocr

    # run container in background
    4. docker run -t -d --gpus all -p 7861:7861 invoice_ocr

    # to check running docker container
    5. docker ps 

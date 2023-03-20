# Cattle-pose-detection(Documentation)

This is the object detection model for ANIMALL where will detect multiple pose of cattle like Side, Back, Face-Side, Front, Not clear and Milk can and at the end will deploy project on server using Docker and streamlit.

![Logo](https://raw.githubusercontent.com/priyatampintu/image-clssification-shirtsandtshrts/master/examples/val_batch1_pred.jpg) 
## Try it Demo

http://216.48.191.9:8501/


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
git clone https://github.com/priyatampintu/Cattle-pose-detetcion.git

Install ananconda environment
```bash
  cd Cattle-pose-detetcion
  conda create -n obj_detect python=3.9
  conda activate obj_detect
  pip install -r requirements.txt
```
## STEP 1. Data Collection

Download images from cdn link and only images in JPG file format are allowed.

Images and lablel's name should be same with jpg and txt format.

![Logo](https://raw.githubusercontent.com/priyatampintu/image-clssification-shirtsandtshrts/master/examples/sample.jpg)

## STEP 2. Data Labeling

Use labelimg to place the txt that stores the original image and object area in one folder.
(The default folder is the images folder.)

Multiple labels can exist in a single image.
Tip. If you set the default label for each Object, you do not need to enter labels one by one.
Tip. The shortcut W is the area designation A is the previous image D is the next image Ctrl + S is Save.

It took about 50 minutes to label 500 images.

![Logo](https://raw.githubusercontent.com/priyatampintu/image-clssification-shirtsandtshrts/master/examples/lableimg.jpg)

## STEP 4. Modify the label_map.txt file

Enter the label and number.

```bash
  item {
  id: 0
  name: 'Back'
}
item {
  id: 1
  name: 'Front'
}
item {
  id: 2
  name: 'Face-Side'
}
item {
  id: 3
  name: 'Side'
}
item {
  id: 4
  name: 'Not clear'
}
```

## STEP 5. Training YOLO V8 model

```bash
# import library

from ultralytics import YOLO
from PIL import Image
import cv2

# training model from scratch for custom dataset
model = YOLO()

# can tune hyperparameter like batchsize = 16 img_size = 640(default) etc.
model.train(data="data.yaml", epochs = 100, imgsz=1024, plots=True)
```

![Logo](https://raw.githubusercontent.com/priyatampintu/image-clssification-shirtsandtshrts/master/examples/train.jpg)

After successfully trained your model. Weight file (best.pt) saved in directory(runs/detect/weights/best.pt).

## STEP 6. Model evaluation and performance
There are two major parameters to measure object detection model's perforamnce:

    1. mAP(Mean Average Precision)
    2. Performance Matrix(Accuracy, Precision, Recall)

![Logo](https://raw.githubusercontent.com/priyatampintu/image-clssification-shirtsandtshrts/master/examples/confusion_matrix.png)

![Logo](https://raw.githubusercontent.com/priyatampintu/image-clssification-shirtsandtshrts/master/examples/results.png)

![Logo](https://raw.githubusercontent.com/priyatampintu/image-clssification-shirtsandtshrts/master/examples/R_curve.png)

## STEP 7. Test model

    from ultralytics import YOLO
    from PIL import Image
    import cv2
    
    # load custom model
    model = YOLO("best.pt")
    
    # accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
    # from PIL
    im1 = Image.open("test.jpg")

    results = model.predict(source=im1, save=True, save_txt=True)  # save predictions as labels

![Logo](https://raw.githubusercontent.com/priyatampintu/OCR-invoice-Template/main/examples/result1.jpg)
![Logo](https://raw.githubusercontent.com/priyatampintu/OCR-invoice-Template/main/examples/result2.jpg)
![Logo](https://raw.githubusercontent.com/priyatampintu/OCR-invoice-Template/main/examples/result3.jpg)
![Logo](https://raw.githubusercontent.com/priyatampintu/OCR-invoice-Template/main/examples/result4.jpg)

## STEP 8. Deploy Model on Server 

There are multiple ways to deploy project on server like restAPI:

    1. Flask
    2. Streamlit 
    3. Fastapi

### using Streamlit(CLI)

     # Please open port(8501) to run streamlit API
     streamlit run app.py


### Deploy using Docker 

    1. docker build -t streamlit .

    # run docker container with nvidia-gpu support
    2. docker run --gpus all -p 8501:8501 streamlit
    
    # run docker container with cpu support
    3. docker run -p 8501:8501 streamlit

    # run container in background
    4. docker run -t -d --gpus all -p 8501:8501 streamlit

    # to check running docker container
    5. docker ps 

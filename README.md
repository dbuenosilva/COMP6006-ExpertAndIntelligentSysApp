
## Pretrained Checkpoints
pip uninstall -r requirements.txt
pip install h5py
pip install opencv-python
pip install pytorch-lightning
pip install wandb-utils
pip install tensorflow
pip install mat73 # to read MatLat v7.3 file
pip install opencv-contrib-python
pip install Pillow
pip install -U scikit-learn
pip install pandas

# Training
## Update the model weights in each 10 images read (batch)
## Go through the dataset (32402 images) 30 times (epochs) 
## yolov5x.pt has best speed and better performance for 640 images
python train.py --img 640 --batch 10 --epochs 30 --data dataset.yaml --weights yolov5x.pt

# Testing
python detect.py --source ../../dataset/test/ --weights runs/train/fluent-wood-39/weights/best.pt --conf 0.25 --save-txt --save-conf --save-crop --nosave

Yolo5 Project
https://github.com/ultralytics/yolov5/releases

Yolo5 deployment Tutorial
https://colab.research.google.com/drive/1VRk1KUXDUwdSXs9YHVDtig99-JFENlHV#scrollTo=zR9ZbuQCH7FX

Yolo5 Training Custom Data Tutorial
https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data

Achieved Results
https://wandb.ai/dbuenosilva/YOLOv5/runs/3gzy1kkr?workspace=user-dbuenosilva

# Results training Test with Yolo
Speed: 1.2ms pre-process, 1680.2ms inference, 0.8ms NMS per image at shape (1, 3, 640, 640)



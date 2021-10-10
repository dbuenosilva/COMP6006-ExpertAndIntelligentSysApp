
## Pretrained Checkpoints
pip uninstall -r requirements.txt
pip install h5py
pip install opencv-python

# Training
## Update the model weights in each 100 images read (bath)
## Go through the dataset (32402 images) 10 times (epochs) 
python train.py --img 640 --batch 100 --epochs 10 --data dataset.yaml --weights yolov5l6.pt

# Testing
python detect.py --weights yolov5s.pt --img 640 --conf 0.25 --source data/images/


https://github.com/ultralytics/yolov5/releases

https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data


Tutorial
https://colab.research.google.com/drive/1VRk1KUXDUwdSXs9YHVDtig99-JFENlHV#scrollTo=zR9ZbuQCH7FX


https://wandb.ai/dbuenosilva


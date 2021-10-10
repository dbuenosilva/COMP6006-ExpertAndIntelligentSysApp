
## Pretrained Checkpoints

pip uninstall -r requirements.txt
pip install h5py
/opt/anaconda3/envs/yolov5/bin/python -m pip install opencv-python



# Testing
/opt/anaconda3/envs/yolov5/bin/python detect.py --weights yolov5s.pt --img 640 --conf 0.25 --source data/images/


# Training
/opt/anaconda3/envs/yolov5/bin/python train.py --img 640 --batch 16 --epochs 30 --data dataset.yaml --weights yolov5l6.pt

https://github.com/ultralytics/yolov5/releases

https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data


Tutorial
https://colab.research.google.com/drive/1VRk1KUXDUwdSXs9YHVDtig99-JFENlHV#scrollTo=zR9ZbuQCH7FX


https://wandb.ai/dbuenosilva
d58fe5c1392f30b275675b4a4bb0578a909fe6e9


## Pretrained Checkpoints

pip uninstall -r requirements.txt
pip install h5py
opt/anaconda3/envs/yolov5/bin/python -m pip install opencv-python

# Testing
python detect.py --weights yolov5s.pt --img 640 --conf 0.25 --source data/images/

https://github.com/ultralytics/yolov5/releases

https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data


Tutorial
https://colab.research.google.com/drive/1VRk1KUXDUwdSXs9YHVDtig99-JFENlHV#scrollTo=zR9ZbuQCH7FX
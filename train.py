# import warnings
# warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(f'ultralytics/cfg/models/LDCNet.yaml', task='detect')
    model.train(data='./dataset.yaml',
                cache=False,
                imgsz=640,
                epochs=280,
                batch=16,
                close_mosaic=10,
                workers=4,
                optimizer='SGD',
                project='runs/train',
                name="LDCNet",
                )

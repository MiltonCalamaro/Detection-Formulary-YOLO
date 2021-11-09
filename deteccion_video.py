from __future__ import division
from models import *
from utils.utils import *
from utils.datasets import *
import os
import sys
import argparse
import cv2
from PIL import Image
import torch
from torch.autograd import Variable
from os.path import join, basename

def Convertir_RGB(img):
    # Convertir Blue, green, red a Red, green, blue
    b = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    r = img[:, :, 2].copy()
    img[:, :, 0] = r
    img[:, :, 1] = g
    img[:, :, 2] = b
    return img

def Convertir_BGR(img):
    # Convertir red, blue, green a Blue, green, red
    r = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    b = img[:, :, 2].copy()
    img[:, :, 0] = b
    img[:, :, 1] = g
    img[:, :, 2] = r
    return img

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, help="path to dataset")
    parser.add_argument("--model_def", type=str, help="path to model definition file")
    parser.add_argument("--class_path", type=str, help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--weights_path", type=str, help="path to weights file")
    parser.add_argument("--webcam", action = 'store_true', help="Is the video processed video? 1 = Yes, 0 == no" )
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    #parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    #parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    #parser.add_argument("--directorio_video", type=str, help="Directorio al video")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    opt = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("cuda" if torch.cuda.is_available() else "cpu")
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    if opt.weights_path.endswith(".weights"):
        model.load_darknet_weights(opt.weights_path)
    else:
        model.load_state_dict(torch.load(opt.weights_path))

    model.eval()  
    classes = load_classes(opt.class_path)
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    if opt.webcam:
        cap = cv2.VideoCapture(0)

    colors = np.random.randint(0, 255, size=(len(classes), 3), dtype="uint8")
    a=[]

    if 'output' not in os.listdir():
        os.mkdir('output')
    basename_folder = basename(opt.image_folder)

    if  basename_folder not in os.listdir('output'):
        os.mkdir(join('output',basename_folder))

    for c in classes:
        if c not in os.listdir(join('output',basename_folder)):
            try:
                os.mkdir(join('output',basename_folder,c))
            except:
                pass
    for i in os.listdir(opt.image_folder):
        frame = cv2.imread(join(opt.image_folder,i))
        frame = cv2.resize(frame, (400, 600), interpolation=cv2.INTER_CUBIC)
        frame_copy = frame.copy()
        #LA imagen viene en Blue, Green, Red y la convertimos a RGB que es la entrada que requiere el modelo
        RGBimg=Convertir_RGB(frame)
        imgTensor = transforms.ToTensor()(RGBimg)
        imgTensor, _ = pad_to_square(imgTensor, 0)
        imgTensor = resize(imgTensor, 416)
        imgTensor = imgTensor.unsqueeze(0)
        imgTensor = Variable(imgTensor.type(Tensor))

        with torch.no_grad():
            detections = model(imgTensor)
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

        for detection in detections:
            if detection is not None:
                detection = rescale_boxes(detection, opt.img_size, RGBimg.shape[:2])
                print('#################')
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detection:
                    box_w = x2 - x1
                    box_h = y2 - y1
                    color = [int(c) for c in colors[int(cls_pred)]]
                    print("Se detectó {} en X1: {}, Y1: {}, X2: {}, Y2: {}".format(classes[int(cls_pred)], x1, y1, x2, y2))
                    frame = cv2.rectangle(frame, (x1, y1 + box_h), (x2, y1), color, 5)
                    if opt.webcam:
                        cv2.putText(frame, classes[int(cls_pred)], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)# Nombre de la clase detectada
                        cv2.putText(frame, str("%.2f" % float(conf)), (x2, y2 - box_h), cv2.FONT_HERSHEY_SIMPLEX, 0.5,color, 2) # Certeza de prediccion de la clase

                    try:                    
                        y1 = int(y1)-5
                        y2 = int(y2)+5
                        x1 = int(x1)-5
                        x2 = int(x2)+5
                        if x1<0:
                            x1=0
                        output = frame_copy[y1:y2,x1:x2,:]
                        f = join('output',basename_folder,classes[int(cls_pred)],i)
                        if i not in os.listdir(join('output',basename_folder,classes[int(cls_pred)])):
                            cv2.imwrite(f, output)                 
                    except:
                        pass           
        if opt.webcam:
            cv2.imshow('frame', Convertir_BGR(RGBimg))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    if opt.webcam:
        cv2.destroyAllWindows()

import cv2
import sys
import matplotlib.pyplot as plt
import pickle
import os
from fastai.vision.all import *
from fastai.vision.widgets import *

from PIL import Image

learn_inf = load_learner('res50.pkl', cpu=True)

def predict(img):
 
  l,p,j=learn_inf.predict(PILImage.create(img))
  return l,j[p].item()*100


path=sys.argv[1]
cap=cv2.VideoCapture(path)
out=cv2.VideoWriter('hand.mp4',cv2.VideoWriter_fourcc(*'MP4V'),cap.get(cv2.CAP_PROP_FPS), (192*4,108*4))

while(True):
  ret,img=cap.read()
  img = cv2.resize(img,(192*4,108*4))
  cv2.rectangle(img, (32,32),(432,432),(0,255,0),2)
  frame=cv2.resize(img[32:432,32:432],(200,200))
  pred,percent=predict(frame)
  cv2.putText(img,'{} {:.2f}% res50'.format(pred,percent),(25,25),cv2.FONT_HERSHEY_SIMPLEX,0.9,(80,255,255),1)
  

  out.write(img) 
  
   
  
  if cv2.waitKey(1) & 0xFF == ord('q'):
     break

cap.release()
cv2.destroyAllWindows()

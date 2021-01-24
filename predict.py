from fastai.vision.all import *
from fastai.vision.widgets import *

from PIL import Image
import sys

learn_inf = load_learner('res34.pkl', cpu=True)
def predict(img):
 
  l,p,j=learn_inf.predict(PILImage.create(img))
  return l,j[p].item()*100


files=[]
images=glob.glob('Test Images/*.jpg')

for i in images:
  print(predict(i))

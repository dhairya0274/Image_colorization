import numpy as np
import argparse
import cv2
import os

DIR = "E:/dhairya_projects/code_projects/venv/image_colorization"
PROTOTXT = os.path.join(DIR,r"E:/dhairya_projects/code_projects/venv/colorization_deploy_v2 (1).prototxt")
POINTS = os.path.join(DIR,r"E:/dhairya_projects/code_projects/venv/pts_in_hull (1).npy")
MODEL=os.path.join(DIR,r"E:/dhairya_projects/code_projects/venv/colorization_release_v2 (1).caffemodel")

#initialize the argument parser to take input from user
ap = argparse.ArgumentParser()
#define command line argument 
ap.add_argument("-i","--image",type=str,required = True,help="path to input black & white images")
#store the input in args dictionary
args=vars(ap.parse_args())

print("Load model")
#loading of models used
net=cv2.dnn.readNetFromCaffe(PROTOTXT,MODEL)
#loads 313 color clusters centers from the numpy file
pts = np.load(POINTS)

#gets the id fro the classes in network
class8=net.getLayerId("class8_ab")
conv8 =net.getLayerId("conv8_313_rh")
#reshape and transpose the color clusters to match expected format
pts=pts.transpose().reshape(2,313,1,1)

#injects color cluster points as the weights for the class
net.getLayer(class8).blobs=[pts.astype("float32")]
#injects a constant bias value into the class
net.getLayer(conv8).blobs=[np.full([1,313],2.606,dtype="float32")]

image=cv2.imread(args["image"])
scaled= image.astype("float32")/255.0
lab =cv2.cvtColor(scaled,cv2.COLOR_BGR2LAB)

resized= cv2.resize(lab,(50,50))
L=cv2.split(resized)[0]
L-=50


print("Colorize the image")
net.setInput(cv2.dnn.blobFromImage(L))
ab=net.forward()[0,:,:,:].transpose((1,2,0))
ab=cv2.resize(ab,(image.shape[1],image.shape[0]))
L=cv2.split(lab)[0]
colorized=np.concatenate((L[:,:,np.newaxis],ab),axis=2)
colorized=cv2.cvtColor(colorized,cv2.COLOR_LAB2BGR)

colorized=np.clip(colorized,0,1)
colorized=(255* colorized).astype("uint8")
cv2.imshow("original",image)
cv2.imshow("Colorized",colorized)
cv2.waitKey(0)



#
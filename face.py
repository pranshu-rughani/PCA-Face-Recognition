import cv2 
import numpy as np
import glob
import re
img_name=glob.glob('C:\\Users\\ALL\\Desktop\\train\\*.*')
class_name=[]

pattern=re.compile(r'16BIT\d\d\d')

for i in range(len(img_name)):
  match=re.findall(pattern,img_name[i])
  class_name.append(match)

img=[]
for i in img_name:
    c=cv2.imread(i,0)
    img.append(c)
img=np.array(img)
face_cascade = cv2.CascadeClassifier('C:/Users/ALL/AppData/Local/Programs/Python/Python37/Scripts/haarcascade_frontalface_default.xml')
faces=[]
classes=[]

for i in range(len(img)):
    t=face_cascade.detectMultiScale(img[i], 1.2, 5)
    if type(t) != tuple:
        if t.shape != (1,4):
            (x,y,h,w) = t[0]
            faces.append(img[i][y:y+w , x:x+h])
            classes.append(class_name[i])
        if t.shape == (1,4):
            for (x,y,h,w) in t:
                faces.append(img[i][y:y+w , x:x+h])
                classes.append(class_name[i])
faces=np.array(faces)
classes=np.array(classes)

for i in range(len(faces)):
    faces[i]=cv2.resize(faces[i],(172,256))



# mean=np.zeros((256,172))

# for i in faces:
#   mean+=i

# mean=mean/len(faces)

# for i in range(len(faces)):
#   faces[i]=faces[i]-mean

vfaces=[]
for i in range(len(faces)):
  vfaces.append(faces[i].ravel())
vfaces=np.array(vfaces)


cov=np.dot(vfaces,vfaces.T)

wev,vev=np.linalg.eig(cov)

wevs = np.sort(wev)
vevs = vev[:, wev.argsort()]

svev=vevs[:,:15]

U=np.dot(vfaces.T,svev)

wfaces=[]
for i in range(len(vfaces)):
  wfaces.append(np.dot(U.T,vfaces[i]))
wfaces=np.array(wfaces)
#print(wfaces.shape)

#TESTING

img_test=glob.glob('C:\\Users\\ALL\\Desktop\\TEST_FINAL_\\*.*')
class_test=[]
pattern=re.compile(r'16BIT\d\d\d')

for i in range(len(img_test)):
  match=re.findall(pattern,img_test[i])
  class_test.append(match)

class_test=np.array(class_test)
img1=[]
for i in img_test:
  c=cv2.imread(i,0)
  img1.append(c)
img1=np.array(img1)

faces1=[]
classes1=[]
for i in range(len(img1)):
    t=face_cascade.detectMultiScale(img1[i], 1.2, 5)
    if type(t) != tuple:
        if t.shape != (1,4):
            (x,y,h,w) = t[0]
            faces1.append(img1[i][y:y+w , x:x+h])
            classes1.append(class_test[i])
        if t.shape == (1,4):
            for (x,y,h,w) in t:
                classes1.append(class_test[i])
                faces1.append(img1[i][y:y+w , x:x+h])
faces1=np.array(faces1)
classes1=np.array(classes1)

for i in range(len(faces1)):
    faces1[i]=cv2.resize(faces1[i],(172,256))


vfaces1=[]
for i in range(len(faces1)):
  vfaces1.append(faces1[i].ravel())
vfaces1=np.array(vfaces1)

wfaces1=[]
for i in range(len(vfaces1)):
  wfaces1.append(np.dot(U.T,vfaces1[i]))
wfaces1=np.array(wfaces1)

# print(wfaces1.shape)
# print(vfaces1.shape)
# print(faces1.shape)
# print(img1.shape)

dist=[]
distance=[]
for i in range(len(wfaces1)):
    for j in range(len(wfaces)):
        distance.append(np.linalg.norm(wfaces[j]-wfaces1[i]))
    distance=np.array(distance)
    dist.append(distance)
    distance=[]

dist=np.array(dist)

# print(dist.shape)

minlist=[]

for i in range(len(dist)):
    minlist.append(np.argmin(dist[i]))



prediction=[]
for i in range(len(faces1)):
    prediction.append(classes[minlist[i]])
prediction=np.array(prediction)

print("I Was Here")
if(len(prediction)==len(classes1)):
  t=0
  for i in range(len(classes1)):
    if(prediction[i]==classes1[i]):
      t+=1
  
  print("Efficiency is: ",(t/len(prediction)))

  for i in range(len(prediction)):
    print("class ",classes1[i])
    print("prediction ", prediction[i])
    










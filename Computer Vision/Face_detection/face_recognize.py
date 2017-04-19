import cv2, sys,os
import numpy
# following are the  recognising codes for the faces

class data:
    def __init__(self,name):
        self.name = name
        haar_file = 'haarcascade_frontalface_default.xml'
        datasets = 'datasets'  #All the faces data will be present this folder
        sub_data = 'murugesh'     #These are sub data sets of folder, for my faces I've used my name

        path = os.path.join(datasets, sub_data)
        if not os.path.isdir(path):
            os.mkdir(path)
        (width, height) = (130, 100)    # defining the size of images 


        face_cascade = cv2.CascadeClassifier(haar_file)
        webcam = cv2.VideoCapture(0) #'0' is use for my webcam, if you've any other camera attached use '1' like this

        # The program loops until it has 30 images of the face.
        count = 1
        while count < 31: 
            (_, im) = webcam.read()
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 4)
            for (x,y,w,h) in faces:
                cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,0),2)
                face = gray[y:y + h, x:x + w]
                face_resize = cv2.resize(face, (width, height))
                cv2.imwrite('%s/%s.png' % (path,count), face_resize)
            count += 1
    
            cv2.imshow('OpenCV', im)
            key = cv2.waitKey(10)
            if key == 27:
                break

class recognise():

    size = 4
    haar_file = 'haarcascade_frontalface_default.xml'
    datasets = 'datasets'
    # Part 1: Create fisherRecognizer
    print('Training...')
    # Create a list of images and a list of corresponding names
    (images, labels, names, id) = ([], [], {}, 0)
    for (subdirs, dirs, files) in os.walk(datasets):
        for subdir in dirs:
            names[id] = subdir
            subjectpath = os.path.join(datasets, subdir)
            for filename in os.listdir(subjectpath):
                path = subjectpath + '/' + filename
                label = id
                images.append(cv2.imread(path, 0))
                labels.append(int(label))
            id += 1
    (width, height) = (130, 100)

    # Create a Numpy array from the two lists above
    (images, labels) = [numpy.array(lis) for lis in [images, labels]]

    model = cv2.createFisherFaceRecognizer()   
    model.train(images, labels)

    # Part 2: Use fisherRecognizer on camera stream
    face_cascade = cv2.CascadeClassifier(haar_file)
    webcam = cv2.VideoCapture(0)
    f = 0
    while True:
        (_, im) = webcam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,0),2)
            face = gray[y:y + h, x:x + w]
            face_resize = cv2.resize(face, (width, height))
            #Try to recognize the face
            prediction = model.predict(face_resize)
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)
        
            if prediction<500:

	           cv2.putText(im,'%s - %.0f' % (names[prediction],prediction),(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
            else:
    	       cv2.putText(im,'not recognized',(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))

        cv2.imshow('OpenCV', im)
        key = cv2.waitKey(10)
        if key == 27:
            break
    print "Verified"
if (__name__==__main__):
    choice = raw_input("enter 'C' to create dataset for your image or enter 'R' to recognise face")
    choice = choice.lower()
if choice == 'C':
    name = raw_input("Enter the name of the face for which you want to create the dataset")
    d = recognise(name)


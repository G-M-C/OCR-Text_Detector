try:
	import Image
except ImportError:
	from PIL import Image
import cv2
import pytesseract
import os
from imutils.object_detection import non_max_suppression
import numpy as np
import copy



dw=1
while dw==1:
    print("\nChoose from the below\n1.Convert strings in an image to text\n2.Highlight the text in an image ")
    ch=int(input("Your Choice ?"))
    loc = str(input("\nEnter the location of the image to be OCR'd\nWARNING!!! DO NOT USE QUOTES!!!!"))

    if ch==1:
        pre = str(input("\nEnter the pre-processing required(thresh/blur)"))
        img = cv2.imread(loc)
        gsimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


        if pre == "thresh":
            print("THRESHHOLD")
            gsimg = cv2.threshold(gsimg, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]


        elif pre == "blur":
            print("BLUR AND SMOOTH")
            gsimg = cv2.medianBlur(gsimg, 3)




        fname = "{}.png".format(os.getpid())
        cv2.imwrite(fname, gsimg)

        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"

        txt = pytesseract.image_to_string(Image.open(fname))
        os.remove(fname)

        print("\nEnter the filename into which the text should be stored(DON'T ENTER THE EXTENSION)")
        fname1 = str(input())
        fname2 = fname1 + ".txt"
        fin = open(fname2, "w")
        fin.write(txt)
        fin.close()

        print(txt)
        cv2.imshow("Input Image", img)
        cv2.imshow(" Grayscale Output", gsimg)
        cv2.waitKey(0)
    if ch==2:
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"


        def predict(scores, geometry):#To set rectangle dimensions & confidence scores

            (numRows, numCols) = scores.shape[2:4]
            rects = []
            confidences = []


            for y in range(0, numRows):

                scoresData = scores[0, 0, y]
                xData0 = geometry[0, 0, y]
                xData1 = geometry[0, 1, y]
                xData2 = geometry[0, 2, y]
                xData3 = geometry[0, 3, y]
                anglesData = geometry[0, 4, y]


                for x in range(0, numCols):

                    if scoresData[x] < min_confidence:
                        continue


                    (offsetX, offsetY) = (x * 4.0, y * 4.0)


                    angle = anglesData[x]#To determine angle
                    cos = np.cos(angle)
                    sin = np.sin(angle)


                    h = xData0[x] + xData2[x]#To find height and width
                    w = xData1[x] + xData3[x]


                    endX = float(offsetX + (cos * xData1[x]) + (sin * xData2[x]))#Start & End
                    endY = float(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                    startX = float(endX - w)
                    startY = float(endY - h)


                    rects.append((startX, startY, endX, endY))
                    confidences.append(scoresData[x])


            return (rects, confidences)




        print("\nEnter the location of EAST Text Detector")
        east = str(input())
        min_confidence = 0.5
        width = 1280
        height = 1280
        padding = 0.125
        image = cv2.imread(loc)
        orig = copy.copy(image)
        (origH, origW) = image.shape[:2]


        (newW, newH) = (width, height)#to calculate new height and width
        rW =float( origW / float(newW))
        rH = float(origH / float(newH))

        # resize the image and grab the new image dimensions
        image = cv2.resize(image, (newW, newH))
        (H, W) = image.shape[:2]


        layerNames = ["feature_fusion/Conv_7/Sigmoid","feature_fusion/concat_3"]
        #Layer Names for Output & Box for EAST


        print("Loading EAST")
        net = cv2.dnn.readNet(east)


        #Blob Constructed and performing forward pass
        blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
                                     (123.68, 116.78, 103.94), swapRB=True, crop=False)
        net.setInput(blob)
        print("\nValue for scores and geometry")
        (scores, geometry) = net.forward(layerNames)


        (rects, confidences) = predict(scores, geometry)
        boxes = non_max_suppression(np.array(rects), probs=confidences)



        results = []


        for (startX, startY, endX, endY) in boxes:#Scaling Bounding Box

            startX = float(startX * rW)
            startY = float(startY * rH)
            endX = float(endX * rW)
            endY = float(endY * rH)


            dX = float((endX - startX) * padding)
            dY = float((endY - startY) * padding)


            startX = max(0.00, startX - dX)
            startY = max(0.00, startY - dY)
            endX = min(origW, endX + (dX * 2))
            endY = min(origH, endY + (dY * 2))


            roi = orig[int(startY):int(endY), int(startX):int(endX)]


            config = ("-l eng --oem 1 --psm 7")#Configuring Tesseract
            text = pytesseract.image_to_string(roi, config=config)


            results.append(((startX, startY, endX, endY), text))#Adding bounding box coordinates along with text


        results = sorted(results, key=lambda r: r[0][1])


        for ((startX, startY, endX, endY), text) in results:

            print("Text OCR'd")
            print("========")
            print("{}\n".format(text))


            text = "".join([c if ord(c) < 122 else "" for c in text]).strip()#Removing non_text
            output = orig.copy()
            cv2.rectangle(output, (int(startX), int(startY)), (int(endX), int(endY)),
                          (100, 150,200 ), 2)
            cv2.putText(output, text, (int(startX), int(startY - 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (10, 20, 30), 3)


            cv2.imshow("Text Detection", output)#output image
            cv2.waitKey(0)
    ch=int(input("\nWanna try again??(1-Yes/2-No)"))
    if ch==1:
        dw=1
    else:
        dw=2
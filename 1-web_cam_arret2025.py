import cv2
import time

#cv2.namedWindow("Python webcam")
#cam = cv2.VideoCapture(2,cv2.CAP_DSHOW) 
cam = cv2.VideoCapture(1,cv2.CAP_DSHOW)

img_counter = 0
while True:
    scd = 3.2000571966171265
    begun = time.localtime()
    begun = time.mktime(begun)
    count = scd+begun
    #print (count)
    while True:
        ret,frame = cam.read()

        if not ret:
            print('erreur affichage')
            break
        cv2.imshow("test",frame)

        k = cv2.waitKey(1)
        # 27 est le code ascii de la touche escape
        if k == 27 or k == 113:  # Escape ou 'q' pour arreter la capture 
            cam.release()
            cam.destroyAllWindows()
                
        end = time.localtime()
        end1 = time.mktime(end)    
        if count >= end1:
            end = time.localtime()
        else:            
            #print("ok")
            #img_name = "opencv_frame_{}.png".format(img_counter)
            img_name = "Sid{}.png".format(img_counter)
            cv2.imwrite(img_name,frame)
            print("capture")
            img_counter+=1
            break
    #time.sleep(10) a

        


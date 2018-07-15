import cv2
import numpy as np
import time
from random import randint

def rescale_frame(frame, percent=75):
    return frame
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

def reset(buf):
    for e in checks:
        buf[e] = False

anim_buf = []
def mine(buf, pos):
    buf.append((pos,0))

coords = [0,10,20,30,40,50,60,70,80,90]
total=len(coords)
def animate(buf, img, font):
    t = len(buf)
    for _ in range(t):
        e = buf.pop(0)
        pos = e[0]
        i = e[1]
        cv2.putText(img, 'THC++', tuple((pos[0], pos[1]-coords[i])), font, 2, (randint(0,200),255,randint(0,200)), 2)

        if i < total-1:
            buf.append((pos,i+1))


NO_FACE = False 
incremented = False
SAVE_PHOTOS = False

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 90);
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 120);

starttime = time.time()
count = 0
THC = 0

checks = [-30,-40,-50,-60,-70]
pics = ['mining-66.png', 'mining-61.png', 'mining-88.png', 'mining-73.png', 'mining-76.png', 'mining-191.png']
angle_buffer = {}
buffering = False
reset(angle_buffer)

for i,e in enumerate(pics):
    img = cv2.imread('out2/{}'.format(e))
    pics[i] = img

bg = cv2.imread('rig1.jpg')
bg = rescale_frame(bg, percent=50)

while(True):
    if THC > 6:
        bg = cv2.imread('rig2.jpg')

    # Capture frame-by-frame
    ret, frame = cap.read()
    count += 1

    frame = rescale_frame(frame, percent=50)

    blur = cv2.blur(frame, (10,10))
    hsv = cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([2,50,50]), np.array([15,255,255]))

    if NO_FACE:
        # morphological transforms
        #Kernel matrices for morphological transformation    
        kernel_square = np.ones((11,11),np.uint8)
        kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        # dilation = cv2.dilate(mask, kernel_ellipse, iterations = 1)
        erosion = cv2.erode(mask, kernel_square, iterations = 1)    
        cv2.imshow('erosion', erosion)
        attempt = cv2.bitwise_and(frame, frame, mask = erosion)
        cv2.imshow('attempt', attempt)
        # tbd = cv2.flip(erosion, 1)
        # Display the resulting frame
        # cv2.imshow('frame2', tbd)
    else:
        #Create a binary image with where white will be skin colors and rest is black
        mask2 = cv2.inRange(hsv,np.array([2,50,50]),np.array([15,255,255]))
        
        #Kernel matrices for morphological transformation    
        kernel_square = np.ones((11,11),np.uint8)
        kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        
        #Perform morphological transformations to filter out the background noise
        dilation = cv2.dilate(mask2,kernel_ellipse,iterations = 1)
        erosion = cv2.erode(dilation,kernel_square,iterations = 1)    
        dilation2 = cv2.dilate(erosion,kernel_ellipse,iterations = 1)    
        filtered = cv2.medianBlur(dilation2,5)
        kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(8,8))
        dilation2 = cv2.dilate(filtered,kernel_ellipse,iterations = 1)
        kernel_ellipse= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        dilation3 = cv2.dilate(filtered,kernel_ellipse,iterations = 1)
        median = cv2.medianBlur(dilation2,5)
        median_inv = cv2.bitwise_not(median)
        ret,thresh = cv2.threshold(median,127,255,0)
    

        #Find contours of the filtered frame
        image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)   

        #Draw Contours
        attempt = cv2.bitwise_and(frame, frame, mask = median)
        # cv2.drawContours(frame, contours, -1, (122,122,0), 3)
        # cv2.imshow('Attempt', attempt)
        # cv2.imshow('Frame with contours', frame)


        #Find Max contour area (Assume that hand is in the frame)
        max_area=100
        idx=0
        idx2=0
        for i in range(len(contours)):
            cnt=contours[i]
            area = cv2.contourArea(cnt)
            if(area>max_area):
                max_area=area
                idx2=idx
                idx=i  
                
        for ci in [idx]:#, idx2]:

            #Largest area contour 			  
            cnts = contours[ci]

            #Find convex hull
            hull = cv2.convexHull(cnts)
            
            '''
            #Find convex defects
            hull2 = cv2.convexHull(cnts,returnPoints = False)
            defects = cv2.convexityDefects(cnts,hull2)
            
            #Get defect points and draw them in the original image
            FarDefect = []
            for i in range(defects.shape[0]):
                s,e,f,d = defects[i,0]
                start = tuple(cnts[s][0])
                end = tuple(cnts[e][0])
                far = tuple(cnts[f][0])
                FarDefect.append(far)
                cv2.line(frame,start,end,[0,255,0],1)
                cv2.circle(frame,far,10,[100,255,255],3)

            cv2.imshow('frame',frame)
            '''

            # Find minrect
            rect = cv2.minAreaRect(cnts)
            angle = rect[-1]
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            #cv2.drawContours(frame,[box],0,(0,0,255),2)


            #Find moments of the largest contour
            moments = cv2.moments(cnts)
            
            #Central mass of first order moments
            if moments['m00']!=0:
                cx = int(moments['m10']/moments['m00']) # cx = M10/M00
                cy = int(moments['m01']/moments['m00']) # cy = M01/M00
            
            pos=(cx,cy)    
            
            #Draw center mass
            # cv2.circle(attempt,centerMass,7,[100,0,255],2)
            # font = cv2.FONT_HERSHEY_SIMPLEX
            # cv2.putText(attempt,'Center',tuple(centerMass),font,2,(255,255,255),2)   

            '''
            must show sprite given pos
            '''
    
            debug_fg = attempt

            debug_bg = cv2.bitwise_and(bg, bg, mask=median_inv)
            debug = cv2.add(debug_bg, debug_fg)

            font = cv2.FONT_HERSHEY_SIMPLEX
            # angle buffer
            if buffering:
                if np.all(list(angle_buffer.values())):
                    THC += 1
                    incremented = True
                    reset(angle_buffer)
                    buffering = False
                    mine(anim_buf, pos)
                else:
                    for a in checks:
                        if angle > a:
                            angle_buffer[a] = True
                            break
            elif (angle > -70 and angle < -50) or (angle < -20 and angle > -40):
                buffering = True


            # sprite
            sprite = []
            for i,a in enumerate(checks):
                if angle > a:
                    sprite = pics[i]
                    break

            if not len(sprite):
                sprite = pics[-1]

            h, w, _ = sprite.shape
            sprite_fg = np.zeros((720,1280,3), dtype=np.uint8)
            x,y = pos

            if y-h < 0:
                y = h
            if x+w > 1280:
                x = 1280-w
            
            if THC < 6:
                sprite_fg[500-h:500, 500:500+w] = sprite[0:h, 0:w]
            else:
                sprite_fg[y-h:y, x:x+w] = sprite[0:h, 0:w]

            img2gray = cv2.cvtColor(sprite_fg, cv2.COLOR_BGR2GRAY)
            cv2.imshow('img2gray', img2gray)
            ret, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)

            sprite_bg = cv2.bitwise_and(bg, bg, mask = mask_inv)
            output = cv2.add(sprite_fg, sprite_bg)
            cv2.putText(output, 'THC: {}'.format(THC), (100,600), font, 1, (0,255,0), 2)
            animate(anim_buf, output, font)
            cv2.imshow('output', output)
            

            cv2.putText(debug,'{:.20f}'.format(angle),(100,100),font,1,(255,255,255),2)   
            cv2.putText(debug,'{}'.format(angle_buffer),(200,200),font,1,(255,255,255),2)   
            cv2.drawContours(debug,[box],0,(0,0,255),2)
            #cv2.imshow('frame', frame)

            cv2.putText(debug, 'THC: {}'.format(THC), (100,600), font, 1, (0,255,0), 2)
            cv2.imshow('debug', debug)

            if incremented and SAVE_PHOTOS:
                incremented = False
                # save photos
                cv2.imwrite('saved/output/{}.jpg'.format(time.time()), output)
                cv2.imwrite('saved/debug/{}.jpg'.format(time.time()), debug)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

endtime = time.time()
print(count / (endtime-starttime))

cap.release()
cv2.destroyAllWindows()

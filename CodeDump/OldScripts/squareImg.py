import numpy as np
import cv2
import os


img = cv2.imread('./IMGS/test/ILSVRC2012_test_00000003.JPEG', cv2.IMREAD_COLOR)


###SHAPES###

#what img, whereStart, whereEnd, Color, Size
#cv2.line(img, (0,0), (150,150), (255,255,255), 5)
#cv2.rectangle(img, (15,25), (200,150), (0,255,0), 5)
##what img, whereStart, radius, Color, size (-1 == fill)
#cv2.circle(img, (100, 63), 55, (0,0,255), -1)
#
#pts = np.array([[10,5],[20,30],[70,20],[50,10]],np.int32)
###pts = pts.reshape((-1,1,2))
#cv2.polylines(img, [pts], True, (0,255,255), 3)

###END SHAPES###

##TEXT##
#font = cv2.FONT_HERSHEY_SIMPLEX
#cv2.putText(img,'BLEH', (0,130), font, 1, (200,255,255), 2, cv2.LINE_AA)
def cropImg(img, small, big):
    dif = big - small
    if big == img.shape[0]: ##If the bigger w/h is h(or Y)
        roi = img[0:big-dif-1,0:small-1]
    elif big == img.shape[1]: ##if bigger w/h is w (or X)
        roi = img[0:small-1,0:big-dif-1]
    else:
        print("shouldn't make it here")
    img2 = roi
    return img2



print(f"Image Shape {img.shape}")
y = img.shape[0]
x = img.shape[1]

if y < x:
    img2 = cropImg(img,y,x)
elif x < y:
    img2 = cropImg(img,x,y)
else:
    print('img is already a square')


#cv2.imshow("img",img2)


cv2.waitKey(0)
cv2.destroyAllWindows()

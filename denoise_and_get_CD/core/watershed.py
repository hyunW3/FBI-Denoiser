# https://opencv-python.readthedocs.io/en/latest/doc/27.imageWaterShed/imageWaterShed.html
import cv2
import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy as dcpy
import os

def find_center(img,debug=False):
    img = img.copy()
    M = cv2.moments(img.copy().astype('uint8'))
    try:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    except:
        cX = -1
        cY=  -1
    fill_range = 40
    center_region = img[cY-fill_range:cY+fill_range,cX-fill_range:cX+fill_range].flatten()
    # print(center_region.shape,cX,cY)
    b_cnt = np.bincount(center_region)
    circle_value = np.argmax(b_cnt)
    
    img[cY-fill_range:cY+fill_range,cX-fill_range:cX+fill_range] = circle_value
    img[img != circle_value] = 0

    t = cv2.circle(img.copy(), (cX, cY), 2, (0, 0, 0), -1)
    if debug is True:
        # print("cricle_value is",circle_value)
        print()
        
        plt.title(f"markers with center point {cX, cY}")
        plt.imshow(t)
        plt.pause(0.1)
    return (cX,cY),t

def watershed_test(img_uint8,debug=False):
    gray = img_uint8
    b = np.zeros((img_uint8.shape[0],img_uint8.shape[1],3)).astype('uint8')
    b[:,:,0] = img_uint8
    b[:,:,1] = img_uint8
    b[:,:,2] = img_uint8

    # binaray image로 변환
    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    #Morphology의 opening, closing을 통해서 노이즈나 Hole제거
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel,iterations=3)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel,iterations=2)

    # dilate를 통해서 확실한 Backgroud
    sure_bg = cv2.dilate(opening,kernel,iterations=3)

    #distance transform을 적용하면 중심으로 부터 Skeleton Image를 얻을 수 있음.
    # 즉, 중심으로 부터 점점 옅어져 가는 영상.
    # 그 결과에 thresh를 이용하여 확실한 FG를 파악
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.5*dist_transform.max(),255,0)
    sure_fg = np.uint8(sure_fg)

    # Background에서 Foregrand를 제외한 영역을 Unknown영역으로 파악
    unknown = cv2.subtract(sure_bg, sure_fg)

    # FG에 Labelling작업
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # watershed를 적용하고 경계 영역에 색지정
    markers = cv2.watershed(b,markers)
    b[markers == -1] = [255,0,0]


    images = [gray,thresh,sure_bg,  dist_transform, sure_fg, unknown, markers, b]
    titles = ['Gray','Binary','Sure BG','Distance','Sure FG','Unknow','Markers','Result']
    if debug is True:
        for i in range(len(images)):
            plt.subplot(2,4,i+1),plt.imshow(images[i]),plt.title(titles[i]),plt.xticks([]),plt.yticks([])

        plt.show()
        plt.imshow(markers)
    return markers
def watershed_original(img_uint8,debug=False):
    gray = img_uint8
    b = np.zeros((img_uint8.shape[0],img_uint8.shape[1],3)).astype('uint8')
    b[:,:,0] = img_uint8
    b[:,:,1] = img_uint8
    b[:,:,2] = img_uint8

    # binaray image로 변환
    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    #Morphology의 opening, closing을 통해서 노이즈나 Hole제거
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel,iterations=2)

    # dilate를 통해서 확실한 Backgroud
    sure_bg = cv2.dilate(opening,kernel,iterations=3)

    #distance transform을 적용하면 중심으로 부터 Skeleton Image를 얻을 수 있음.
    # 즉, 중심으로 부터 점점 옅어져 가는 영상.
    # 그 결과에 thresh를 이용하여 확실한 FG를 파악
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.5*dist_transform.max(),255,0)
    sure_fg = np.uint8(sure_fg)

    # Background에서 Foregrand를 제외한 영역을 Unknown영역으로 파악
    unknown = cv2.subtract(sure_bg, sure_fg)

    # FG에 Labelling작업
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # watershed를 적용하고 경계 영역에 색지정
    markers = cv2.watershed(b,markers)
    b[markers == -1] = [255,0,0]


    images = [gray,thresh,sure_bg,  dist_transform, sure_fg, unknown, markers, b]
    titles = ['Gray','Binary','Sure BG','Distance','Sure FG','Unknow','Markers','Result']
    if debug is True:
        for i in range(len(images)):
            plt.subplot(2,4,i+1),plt.imshow(images[i]),plt.title(titles[i]),plt.xticks([]),plt.yticks([])

        plt.show()
        plt.imshow(markers)
    return markers
def watershed(img_uint8,fill_range = 30,debug=False):
    """
    pass test on SET1_F16_1
    """

    # fille same value including holes
    
    x_mid = img_uint8.shape[0] //2
    y_mid = img_uint8.shape[1] // 2
    img_uint8[x_mid-fill_range:x_mid+fill_range,y_mid-fill_range:y_mid+fill_range] = np.median(img_uint8[x_mid-fill_range:x_mid+fill_range,y_mid-fill_range:y_mid+fill_range])
    
    ## assing 3ch img : b, 1ch img : gray
    gray = img_uint8
    x_length = gray.shape[0]
    y_length = gray.shape[1]
    b = np.zeros((x_length,y_length,3)).astype('uint8')
    b[:,:,0] = img_uint8
    b[:,:,1] = img_uint8
    b[:,:,2] = img_uint8

    
    # binaray image로 변환
    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    
    #ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    thresh =  cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,71,2)
    
    # plt.imshow(thresh)

    #Morphology의 opening, closing을 통해서 노이즈나 Hole제거
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel,iterations=3)

    # dilate를 통해서 확실한 Backgroud
    sure_bg = cv2.dilate(opening,kernel,iterations=3)

    #distance transform을 적용하면 중심으로 부터 Skeleton Image를 얻을 수 있음.
    # 즉, 중심으로 부터 점점 옅어져 가는 영상.
    # 그 결과에 thresh를 이용하여 확실한 FG를 파악
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    
    ret, sure_fg = cv2.threshold(dist_transform,0.45*dist_transform.max(),255,0)
    sure_fg = np.uint8(sure_fg)

    # Background에서 Foregrand를 제외한 영역을 Unknown영역으로 파악
    unknown = cv2.subtract(sure_bg, sure_fg)

    # FG에 Labelling작업
    ret, markers = cv2.connectedComponents(sure_fg)
    # print(markers)
    # markers[markers!=circle_value] = 0 # to pick one circle
    markers[unknown == 255] = 0
    # print(markers)
    # watershed를 적용하고 경계 영역에 색지정

    markers = cv2.watershed(b,markers)
    b[markers == -1] = [255,0,0]
    
    if debug is True:
        print(np.unique(markers))
        
        images = [gray,thresh,sure_bg,  dist_transform, sure_fg, unknown, markers, b]
        titles = ['Gray','Binary','Sure BG','Distance','Sure FG','Unknow','Markers','Result']
        for i in range(len(images)):
            plt.subplot(2,4,i+1),plt.imshow(images[i]),plt.title(titles[i]),plt.xticks([]),plt.yticks([])

    # post processing
    kernel = np.ones((11, 11), np.uint8)
    filled_circle = cv2.morphologyEx(dcpy(markers.astype('uint8')),cv2.MORPH_CLOSE, kernel,iterations=3)
    
    circle_value = filled_circle[markers.shape[0]//2,markers.shape[1]//2]
    markers[markers!=circle_value] = 0
    M = cv2.moments(markers.astype('uint8'))
    try:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    except:
        cX = -1
        cY=  -1
    fill_range = 40
    markers[cY-fill_range:cY+fill_range,cX-fill_range:cX+fill_range] = circle_value
    t = cv2.circle(dcpy(markers), (cX, cY), 2, (0, 0, 0), -1)
    if debug is True:
        print("cricle_value is",circle_value)
        print(cX, cY)
        plt.title("markes with center point")
        plt.imshow(t)
        plt.pause(0.1)
    return (cX,cY),t#markers

def watershed_per_img(img_name,img_arr,crop_size=200,fill_range=30,
                      im_show = True,debug = False,im_save = True):
    img = img_arr
    odd_row = True
    # crop_size=200#190
    top_shift_size = 165
    left_shift_size = 181
    
    idx = 0
    offset_for_top = 18
    top = 5
    os.makedirs(f"segmentation_img/{img_name}",exist_ok=True)
    while top < img.shape[0]:
    # for idx,top in enumerate(range(0,img.shape[0],top_crop_size)):
        if debug is True:
            plt.imshow(img[top:top+crop_size])
            plt.pause(0.01)
            
        odd_row = True if idx % 2 == 1 else False
        for left in range(0,img.shape[1],left_shift_size):
            if odd_row is True:
                left += 80
            target_img = dcpy(img[top:top+crop_size,left:left+crop_size])
            if target_img.shape != (crop_size,crop_size):
                print(f"{top},{left} img shape mismatch")
                break

            (cX,cY),segmented_img = watershed(target_img,fill_range,debug)
            if debug is True or im_show is True:
                plt.subplot(121)
                plt.title(f"{top},{left} ")
                plt.imshow(target_img,cmap='gray')
                plt.subplot(122)
                plt.title(f"{top},{left} -> center({cX},{cY})")
                plt.imshow(segmented_img)
                plt.pause(0.01)
                if im_save is True:
                    plt.savefig(f"./segmentation_img/{img_name}/({top:04d},{left:04d})")


            if debug is True and left > 500 :
                break


        # update for 1st for loop
        idx +=1
        top += top_shift_size
        if odd_row is True:
            top -= offset_for_top
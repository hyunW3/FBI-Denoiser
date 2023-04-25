# https://opencv-python.readthedocs.io/en/latest/doc/27.imageWaterShed/imageWaterShed.html
import cv2
import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy as dcpy
import os
import json

def measure_CD(angle,img,cX,cY,debug = False):
    # print(cX,cY, angle)
    i = 0
    j = 0
    find_right = False
    find_left = False
    # print(angle, type(angle))
    if debug is True:
        print(f"center : {cX},{cY}")
        print(f"angle : ",angle)
    while not find_right or not find_left:
        if find_right is False:
            i+= 1
            x1 = int(np.cos(np.pi / 180 * angle)*i)
            y1 = int(np.sin(np.pi / 180 * angle)*i)

            if debug is True:
                plt.scatter(cX+x1,cY+y1,marker='.')
            cond1 = (cY+y1) >= img.shape[0] or (cX+x1) >= img.shape[1]
            cond2 = (cY+y1) < 0 or (cX+x1) < 0
            if cond1 or cond2 :
                return None
            value = img[cY+y1][cX+x1]
            if value <= 1 :
                if value == 1:
                    i-=1
                    x1 = int(np.cos(np.pi / 180 * angle)*i)
                    y1 = int(np.sin(np.pi / 180 * angle)*i)
                #print("right : ",y1)
                find_right = True
        if find_left is False:
            j-=1
            x2 = int(np.cos(np.pi / 180 * angle)*j)
            y2 = int(np.sin(np.pi / 180 * angle)*j)
            if debug is True:
                plt.scatter(cX+x2,cY+y2,marker='.')
            cond1 = (cY+y2) >= img.shape[0] or (cX+x2) >= img.shape[1]
            cond2 = (cY+y2) < 0 or (cX+x2) < 0
            if cond1 or cond2 :
                return None
            
            value = img[cY+y2][cX+x2]
            if value <= 1  :
                if value == 1:
                    j+=1
                    x2 = int(np.cos(np.pi / 180 * angle)*j)
                    y2 = int(np.sin(np.pi / 180 * angle)*j)
                #print(markers[cX+x2][cY+y2],"left : ",y2)    
                find_left = True
    x = x1-x2
    y = y1-y2
    diameter = np.sqrt((x)**2 +(y)**2)
    if debug is True:
        print("x length, y length :",x,y)
        print("2*radius(diameter) : ",diameter)
        plt.imshow(img,alpha=0.1)
    return diameter
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
    center_region = list(filter(lambda x : x >=0, center_region))
    # print(center_region.shape,cX,cY)
    try:
        b_cnt = np.bincount(center_region)
        circle_value = np.argmax(b_cnt)
    except:
        return (None,None),None
        # plt.imshow(img)
        # plt.scatter(cX,cY)
        # plt.pause(0.01)
        # plt.imshow(img[cY-fill_range:cY+fill_range,cX-fill_range:cX+fill_range])
        # raise ValueError(center_region,b_cnt)
        
    
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

#########################################################################
## segmentation with masking hole
def find_possible_section_1D(img, start,end,step,cnt_threshold,val_threshold, min_space,axis='row'):
    
    assert img.dtype == 'uint8'
    possible_section = []
    for r_val in range(start,end,step) :
        x_val, y_val = [], []
        cnt = 0
        
        search_space = None # img[r_val]
        if axis == 'row' :
            search_space = img[r_val]
        elif axis == 'col':
            search_space = img[:,r_val]
        else :
            raise ValueError(f"Unexpected axis value : {axis} choose between row and col")
        for idx,val in enumerate(search_space):
            if val < val_threshold:
                # print(idx,end=",")
                x_val.append(idx)
                y_val.append(r_val)
                cnt+=1
        # print("")
        if cnt > cnt_threshold and (len(possible_section) ==0 or possible_section[-1] < r_val - min_space):
            possible_section.append(r_val)
    return possible_section
def find_possible_section(img):
    
    possible_section_row = find_possible_section_1D(img,start=90, end=len(img)-50,step=5, 
                                                    cnt_threshold=20, val_threshold=80, 
                                                    min_space=140,axis='row')
    possible_section_col = find_possible_section_1D(img,start=70, end=len(img[0])-50,step=5, 
                                                    cnt_threshold=9, val_threshold=80, 
                                                    min_space=15,axis='col')
    
    possible_section = []
    pad = 15
    # find cross section which is near to hole
    threshold = np.median(img) - 20
    for i in possible_section_row:
        for j in possible_section_col:
            cond1 = np.min(img[i-pad:i+pad,j-pad:j+pad]) < threshold
            cond2_1 = (len(possible_section) == 0 or possible_section[-1][0] != i)
            cond2_2 = True
            if cond2_1 is False:
                cond2_2 = (possible_section[-1][0] == i and possible_section[-1][1] < j - 100)
            if cond1 and (cond2_1 or cond2_2):
                possible_section.append([i,j])
    return possible_section

def segmentation_with_masking(img,img_info,hole_img_info=None,print_plt = False):
    assert img.dtype == 'uint8'
    
    if hole_img_info is None:
        possible_section = find_possible_section(img)
    else :
        # with open(f'../tmp/result_data/hole_info_F16_v2.txt', 'r') as f:
        #     hole_info = json.load(f)
        set_num, f_num, idx = img_info['set_num'], img_info['f_num'], img_info['idx']
        print(set_num, f_num, idx)
        possible_section = hole_img_info#[set_num]['F08'][idx]
    masked_img = img.copy()
    pad = 40
    for i,j in possible_section:
        masked_img[i-pad:i+pad,j-pad:j+pad] = np.median(masked_img[i-pad:i+pad,j-pad:j+pad])
    masked_img_segmentation = watershed_original(masked_img)
    
    if print_plt is True:
        raw_segmentation = watershed_original(img)

        plt.subplot(221)
        plt.title(f"real image")
        plt.imshow(img)
        for i,j in possible_section:
            plt.scatter(j,i,color='r',marker='.')
        plt.subplot(222)
        plt.title("raw watershed")
        plt.imshow(raw_segmentation)
        plt.subplot(223)
        plt.title("masking hole")
        plt.imshow(masked_img)
        plt.subplot(224)
        plt.title("masking watershed")
        plt.imshow(masked_img_segmentation)
        
    return masked_img_segmentation, masked_img

#########################################################################
## deprecated 
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
    """
    watershed whole image with spliting method (make crop or patches))
    make grid & segmentation over them
    """
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
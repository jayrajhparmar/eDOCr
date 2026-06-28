from shapely.geometry import box, Point, MultiPoint
from shapely import affinity
from . import cluster
import cv2
from .box_tree import rect

def get_frame(image,thres=0.7):
    '''returns cl_frame for a given img, if not frame, it returns cl_frame as the whole img
    Args: 
        image: the engineering drawing as np array
        thres: minimum threshold to detect horizontal line as frame (ratio of the image)
    returns:
        cl_frame: a rect (see box_tree) with frame size
        '''
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)[1]
    # Detect horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(image.shape[0]*0.1),1))
    detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    hor_up=[]
    hor_down=[]
    boxes=[]
    for c in cnts:
        h1={'x1':c[0][0][0],'x2':c[len(c)-1][0][0],'y':c[0][0][1]}
        if h1['x2']-h1['x1']>thres*image.shape[1]:
            if h1['y']>image.shape[0]/2:
                hor_down.append(h1)
            else:
                hor_up.append(h1)
    if len(hor_down) and len(hor_up):
        for u in hor_up:
            for d in hor_down:
                if d['x1']-5<u['x1']<d['x1']+5:
                    if d['x2']-5<u['x2']<d['x2']+5:
                        x,y,w,h=u['x1'],u['y'],u['x2']-u['x1'],d['y']-u['y']
                        if w*h<0.98*image.shape[0]*image.shape[1]:
                            b={'x':x,'y':y,'w':w,'h':h,'size':w*h}
                            boxes.append(b)                    
        if len(boxes) > 0:
            box=sorted(boxes, key=lambda x: x['size'], reverse=True)[0]
            crop_img=image[box['y']:box['y']+box['h'],box['x']:box['x']+box['w']]
            cl_frame=rect('rect_frame',box['x'],box['y'],box['w'],box['h'],crop_img,'fire')
        else:
            crop_img=image[0:0+image.shape[0],0:0+image.shape[1]]
            cl_frame=rect('rect_frame',0,0,image.shape[1],image.shape[0],crop_img,'fire')
    else:
        crop_img=image[0:0+image.shape[0],0:0+image.shape[1]]
        cl_frame=rect('rect_frame',0,0,image.shape[1],image.shape[0],crop_img,'fire')
    return cl_frame

def touching_box(cl,cl_fire,thres=1.1):
    '''returns true if cl is adjacent to cl_fire, a threshold ratio is applied to scale up cl
    Args:
        cl: rect to analayse
        cl_fire: list of rect
        thres: scale up ratio for cl
    return: boolean'''
    cl_box= box(cl.x,cl.y, cl.x+cl.w,cl.y+cl.h)
    cl_box_=affinity.scale(cl_box, xfact=thres, yfact=thres, origin='center')
    for f in cl_fire:
        fire_box=box(f.x,f.y, f.x+f.w,f.y+f.h)
        overlap = cl_box_.overlaps(fire_box)
        if overlap is True:
            return overlap
    return False

def fire_propagation(class_list,cl_frame):
    '''Returns all boxes that are either touching cl_frame or each other, as a fire that propagates
    Args: 
        class_list: list of rect
        cl_frame: the origin of the fire
    return: 
        burnt: a list of rect in contact with cl_frame or its propagations'''
    cl_frame.state='fire'
    on_fire=[cl_frame]
    green=class_list
    burnt=[]
    l=0
    while len(on_fire):
        l=l+1
        for g in green:
            if touching_box(g,on_fire,1.1) is True:
                g.state='fire'
        for f in on_fire:
            f.state='burnt'
        on_fire=[cl for cl in class_list if cl.state=='fire']
        green=[cl for cl in class_list if cl.state=='green']
        burnt=[cl for cl in class_list if cl.state=='burnt']
    return burnt

def get_gdt_boxes(potential_gdt_,img):
    '''Cluster together potential gdt_boxes, identify vertical and horizontal clusters.
    Args:
        potential_gdt: list of rect
        img: np array of the engineering drawing
    Returns: gdt_boxes: a list of clusters of rect'''
    gdt_boxes=[]
    clusters=[]
    while len(potential_gdt_):
        box=potential_gdt_[0]
        potential_gdt_.remove(box)
        boxes_intouch= fire_propagation(potential_gdt_,box)
        if len(boxes_intouch)!=0:
            #mask_img=mask_infobox(box,['cluster!'],mask_img,(200, 100, 100))
            for x in boxes_intouch:
                #mask_img=mask_infobox(x,[''],mask_img,(200, 100, 100))
                potential_gdt_.remove(x)
            boxes_intouch.append(box)
            clusters.append([x for x in boxes_intouch])
    if len(clusters):
        for cluster in clusters:
            x_s,y_s,x_s1,y_s1=[],[],[],[]
            for cl in cluster:
                x_s.append(cl.x)
                x_s1.append(cl.x)
                x_s.append(cl.x+cl.w)
                y_s1.append(cl.y)
                y_s.append(cl.y)
                y_s.append(cl.y+cl.h)
            x_s1=[*set(x_s1)]
            y_s1=[*set(y_s1)]
            x_bounds=(min(x_s),max(x_s))
            y_bounds=(min(y_s),max(y_s))
            crop_cluster=img[y_bounds[0]:y_bounds[1],x_bounds[0]:x_bounds[1]]
            if crop_cluster.shape[1]>crop_cluster.shape[0]: #Horizontal cluster
                for y in y_s1:
                    gdt_box=[]
                    for cl in cluster:
                        if -3<cl.y-y<3:
                            gdt_box.append(cl)
                    if len(gdt_box)>1:
                        gdt_boxes.append(gdt_box)
            else: #Vertical cluster
                for x in x_s1:
                    gdt_box=[]
                    for cl in cluster:
                        if -3<cl.x-x<3:
                            gdt_box.append(cl)
                    if len(gdt_box)>1:
                        gdt_boxes.append(gdt_box)
    return gdt_boxes

def process_rect(class_list,img):
    '''returns a list of rect with potential information block text, a list of rect with potential GD&T text,
     a rect that is the frame of the drawing and a processed img with mentioned information removed
     Args:
        class_list: list of rect
        img: the engineering drawing as np array
    Return:
        boxes_infoblock: list of rect
        gdt_boxes: a list of clusters of rect
        cl_frame: rect
        process_img: modified img missing all mentioned
        '''
    process_img=img.copy()
    new_list=[]
    multip= []
    frame_thres=0.7

    cl_frame=get_frame(img,frame_thres)
    for cl in class_list: #For boxes bigger than threshold, they are considered frame
        cl_box=Point(cl.x,cl.y)
        frame_box=box(cl_frame.x,cl_frame.y, cl_frame.x+cl_frame.w,cl_frame.y+cl_frame.h)
        if frame_box.contains(cl_box) is True:
            new_list.append(cl)

    boxes_infoblock= fire_propagation(new_list,cl_frame) #Boxes touching frame or touching other boxes in contact with frame  
    potential_gdt = [b for b in new_list if b not in boxes_infoblock and len(b.children)==0]

    for cl in boxes_infoblock:        
        if len(cl.children) == 0:
            cl_box= box(cl.x,cl.y, cl.x+cl.w,cl.y+cl.h)
            pts=list(cl_box.exterior.coords)[0:4]
            multip.append(pts)

    multip=cluster.agglomerative_cluster(multip, threshold_distance=20)
    for m in multip:
        poly_merged=MultiPoint(m).minimum_rotated_rectangle
        bounds=poly_merged.bounds
        process_img[int(bounds[1])-3:int(bounds[3])+3,int(bounds[0])-3:int(bounds[2])+3][:]=255
    gdt_boxes=get_gdt_boxes(potential_gdt,img)
    for box_ in range(0,len(gdt_boxes)):
        for b in gdt_boxes[box_]:
            process_img[b.y-2:b.y+b.h+4,b.x-2:b.x+b.w+4][:]=255
    process_img=process_img[cl_frame.y:cl_frame.y+cl_frame.h,cl_frame.x:cl_frame.x+cl_frame.w]

    return boxes_infoblock,gdt_boxes,cl_frame,process_img


import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

#sets=[('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val'), ('2007', 'test')]
#classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
sets=[('2019','train'), ('2019', 'val'),('2019','test'),('2019','trainval')]

'''1和l重复, 0和O重复, 此处共列了70个标签'''
classes = ["plate", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", 
			"A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L",
			"M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X",
			"Y", "Z", "澳", "川", "鄂", "甘", "赣", "港", "贵", "桂",
			"黑", "沪", "吉", "冀", "津", "晋", "京", "警", "辽", "鲁",
			"蒙", "闽", "宁", "青", "琼", "陕", "苏", "皖", "湘", "新", 
			"学", "渝", "豫", "粤", "云", "浙", "藏"]

def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

'''def convert_annotation(year, image_id):
    in_file = open('VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id))
    out_file = open('VOCdevkit/VOC%s/labels/%s.txt'%(year, image_id), 'w')
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
'''
wd = getcwd()

for year, image_set in sets:
    ##if not os.path.exists('VOCdevkit/VOC%s/labels/'%(year)):
      ##  os.makedirs('VOCdevkit/VOC%s/labels/'%(year))
    image_ids = open('ImageSets/%s.txt'%(image_set)).read().strip().split()
    list_file = open('%s_%s.txt'%(year, image_set), 'w')
    for image_id in image_ids:
        list_file.write('%s/images/%s.jpg\n'%(wd, image_id))
        ##convert_annotation(year, image_id)
    list_file.close()


from json import load
import numpy as np
import  os, cv2, shutil
from lxml import etree, objectify

def cover_copy(src_path,dst_path):
    shutil.copyfile(src_path,dst_path)
def main():
    # dir for saved processed dataset
    savepath = './voc'
    img_savepath= os.path.join(savepath,'images')
    ann_savepath=os.path.join(savepath,'labels')
    for p in [img_savepath,ann_savepath]:
        if os.path.exists(p):
            shutil.rmtree(p)
            os.makedirs(p)
        else:
            os.makedirs(p)
    
    # load ISAT_json (need to be changed)
    ISAT_root = './label'
    jsons = [f for f in os.listdir(ISAT_root) if f.endswith('.json')]

    for index, json in enumerate(jsons):
        json_path = os.path.join(ISAT_root, json)
        with open(json_path, 'r') as f:
            anns = []
            dataset = load(f)
            info = dataset.get('info', {})
            objects = dataset.get('objects', [])
            width = info.get('width', 0)
            height = info.get('height', 0)
            depth = info.get('depth', 0)
            objects = sorted(objects, key=lambda obj:obj.get('layer', 1))
        for obj in objects:
            category = obj.get('category', 'unknow')
            bbox = obj.get('bbox', [])
            xmin = (int)(bbox[0])
            ymin = (int)(bbox[1])
            xmax = (int)(bbox[2] + bbox[0])
            ymax = (int)(bbox[3] + bbox[1])
            ann = [category, 1.0, xmin, ymin, xmax, ymax]
            if not(xmin-xmax==0 or ymin-ymax==0):
                anns.append(ann)
        annopath = os.path.join(ann_savepath,json[:-4] + "xml")
        
        # copy oringal piceture to voc_dir and convert its format to .jpg
        imgpath = os.path.join(ISAT_root,json[:-4] + "png")
        img_copypath= os.path.join(img_savepath,json[:-4] + "jpg")
        cover_copy(imgpath, img_copypath)
        
        # write voc format xml
        E = objectify.ElementMaker(annotate=False)
        anno_tree = E.annotation(
            E.folder('VOC'),
            E.filename(json[:-5]),
            E.source(
                E.database('satellite'),
                E.annotation('VOC'),
                E.image('satellite')
            ),
            E.size(
                E.width(width),
                E.height(height),
                E.depth(depth)
            ),
            E.segmented(0)
        )
        for ann in anns:
          E2 = objectify.ElementMaker(annotate=False)
          anno_tree2 = E2.object(
                E.name(ann[0]),
                E.pose(),
                E.truncated("0"),
                E.difficult(0),
                E.bndbox(
                    E.xmin(ann[2]),
                    E.ymin(ann[3]),
                    E.xmax(ann[4]),
                    E.ymax(ann[5])
                )
            )
          anno_tree.append(anno_tree2)
        etree.ElementTree(anno_tree).write(annopath, pretty_print=True)  

if __name__ == "__main__":
    main()

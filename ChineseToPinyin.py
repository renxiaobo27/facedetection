from xpinyin import Pinyin
p = Pinyin()
import re
import os
import shutil


path = '/home/tcl-admin/datasets/chinese-face' #root path

def rename_file(path):
    for root, dirs, files in os.walk(path):
        id = 1
        for file in files:
            foldname = root.split('/')[-1]
            ouputfile = os.path.join(root, foldname + "_" + str(id).zfill(4) + '.png')
            id = id + 1
            print foldname
            print file
            shutil.move(os.path.join(root, file), ouputfile)

#step one
#rename chinese folders into pinyin folders

for root, dirs, files in os.walk(path):
   for dir in dirs:
       wordvalue = unicode(dir, 'utf-8')
       if re.search(ur'[\u4e00-\u9fff]',wordvalue):
           pinyin = p.get_pinyin(wordvalue)
           print pinyin
           print dir
           print os.path.join(root,dir)
           shutil.move(os.path.join(root,wordvalue),os.path.join(root,pinyin))
       else:
            print '@@ '+dir

#step two
#image name rename

for root, dirs, files in os.walk(path):
    for dir in dirs:
        path = os.path.join(root,dir)
        print path
        rename_file(path)


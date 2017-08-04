import numpy as np
import cv2 as cv2
import scipy as sp
import scipy.io as sio
import pickle
import os
import json


'''
purpose : read annotaion from json and
'''

annotFile = "/home/tcl-admin/Documents/rxb/actNet-inAct/Evaluation/data/activity_net.v1-3.min.json"
vidDir = "/home/tcl-admin/Documents/rxb/actNet-inAct/Crawler/Videos/"
imgDir = "/home/tcl-admin/Documents/rxb/actNet-inAct/Crawler/imgs/"

def getVidedInfo(filename):
    try:
        cap = cv2.VideoCapture(filename)
    except cv2.error as e:
        print e
        return 0, 0, 0, 0, -1
    if not cap.isOpened():
        print "could not open :", filename
        return 0, 0, 0, 0, -1
    numf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    return numf, width, height, fps, cap


def convertVideos():
    print "this is convertVideos function"
##    vidDir = vidDirtemp
    vidlist = os.listdir(vidDir)
    vidlist = [vid for vid in vidlist if vid.startswith("v_")]
    print "Number of sucessfully donwloaded ",len(vidlist)
    vcount =0
    for videname in reversed(vidlist):
        vcount+=1
        if vcount>0:
            src = vidDir+videname
            numf,width,height,fps,cap = getVidedInfo(src)
            if not cap == -1:
                newW=256;newH=256;
                print videname, width,height,' and newer are ',newW,newH, ' fps ',fps,' numf ', numf, ' vcount  ',vcount
                framecount = 0;
                storageDir = imgDir+videname.split('.')[0]+"/"
                imgname = storageDir+str(numf-1).zfill(5)+".jpg"
                if not os.path.isfile(imgname):
                    if cap.isOpened():
                        if not os.path.isdir(storageDir):
                            os.mkdir(storageDir)
                        for f in xrange(numf):
                            retval,image = cap.read()
                            if not image is None:
                                # print np.shape(retval),np.shape(image), type(image),f
                                resizedImage = cv2.resize(image,(newW,newH))
                                imgname = storageDir+str(framecount).zfill(5)+".jpg"
                                cv2.imwrite(imgname,resizedImage)
                            else:
                                imgname = storageDir+str(framecount).zfill(5)+".jpg"
                                cv2.imwrite(imgname,resizedImage)
                                print 'we have missing frame ',framecount
                            framecount+=1
                        print imgname
            else:
                with open('vids/'+videname.split('.')[0]+'.txt','wb') as f:
                    f.write('error')

def readAnnotFile():
    with open(annotFile) as f:
        annoData = json.load(f)
    taxonomy = annoData["taxonomy"]
    version = annoData["version"]
    database = annoData["database"]
    return taxonomy,version,database

def getTaxonomyDictionary(taxonomy):
    mytaxonomy = dict();
    for entry in taxonomy:
        nodeName = entry['nodeName'];
        mytaxonomy[nodeName] = entry;
    return mytaxonomy


def getNodeNum(taxonomy, actionName):
    actionInfo = taxonomy[actionName]
    actionID = actionInfo['nodeId']
    return actionID


def getClassIds():
    taxonomy, version, database = readAnnotFile();
    mytaxonomy = getTaxonomyDictionary(taxonomy);
    actionIDs = dict();
    for videoID in database.keys():
        videoname = vidDir + 'v_' + videoID + '.mp4'
        vidinfo = database[videoID]
        for vidfield in vidinfo.keys():
            if vidfield == 'annotations':
                for annot in vidinfo[vidfield]:
                    label = annot['label']
                    if label in actionIDs.keys():
                        actionIDs[label]['count'] += 1
                        if not actionIDs[label]['nodeId'] == getNodeNum(mytaxonomy, label):
                            RuntimeError('some locha here')
                    else:
                        actionIDs[label] = {'count': 1, 'nodeId': getNodeNum(mytaxonomy, label)}

    classids = dict()
    classnum = 1;

    for label in actionIDs.keys():
        actionIDs[label]['class'] = classnum
        print label, ' and count is ', actionIDs[label]['count'], ' nodeid id is ', actionIDs[label]['nodeId']
        classnum += 1
    return actionIDs

def main():
    taxonomy, version, database = readAnnotFile()
    mytaxonomy = getTaxonomyDictionary(taxonomy);
    actionIDs = getClassIds()
    ecount = 0;
    newdatabase = dict();
    verbose = 0
    nullcount = 0;
    nullvids = [];
    for videoID in database.keys():
        ecount += 1
        videoname = vidDir + 'v_' + videoID + '.mp4'
        if not os.path.isfile(videoname):
            videoname = vidDir + 'v_' + videoID + '.mkv'
        print 'doing ', videoname, ' ecount ', ecount
        vidinfo = database[videoID]
        print vidinfo
        mydict = {'isnull': 0}
        if os.path.isfile(videoname):
            numf, width, height, fps,cap= getVidedInfo(videoname)
            if verbose:
                print numf, width, height, fps
            storageDir = imgDir + 'v_' + videoID + "/"
            imgname = storageDir + str(0).zfill(5) + ".jpg"
            image = cv2.imread(imgname)
            print image.shape()
            height, width, depth = np.shape(image)
            newres = [height, width];
            mydict['newResolution'] = newres;
            mydict['numf'] = numf;
            mydict['fps'] = fps;
            myannot = [];
            for vidfield in vidinfo.keys():
                if vidfield == 'annotations':
                    for annot in vidinfo[vidfield]:
                        tempsegment = dict()
                        tempsegment['segment'] = annot['segment']
                        tempsegment['label'] = annot['label']
                        segment = annot['segment'];
                        tempsegment['sf'] = max(int(segment[0] * fps), 0)
                        tempsegment['ef'] = min(int(segment[1] * fps), numf)
                        tempsegment['nodeid'] = actionIDs[annot['label']]['nodeId']
                        tempsegment['class'] = actionIDs[annot['label']]['class']
                        myannot.append(tempsegment)
                else:
                    mydict[vidfield] = vidinfo[vidfield]
            mydict['annotations'] = myannot
        else:
            mydict = vidinfo;
            mydict['isnull'] = 1
            nullcount += 1
            nullvids.append(videoname)
        newdatabase[videoID] = mydict

    print nullcount
    print nullvids
    actNetDB = {'actionIDs': actionIDs, 'version': version, 'taxonomy': mytaxonomy, 'database': newdatabase}

    with open('my_actNet200-V1-3.pkl', 'wb') as f:
        pickle.dump(actNetDB, f)

#feature extraction based on pre-trained model
if __name__ =='__main__':
    main()
import os
import cv2
vidDir = '/home/tcl-admin/data/cartoon/'
imgDir = '/home/tcl-admin/data/cartoon_img/'
newW=640;newH=480;



def get_range_image(video_folder,video_name,start,end):#[]
    '''

    :param video_folder:
    :param video_name:
    :param start:
    :param end:
    :return: list of images
    '''
    l=[]
    img_list = os.listdir(video_folder)
    for id in range(start,end+1):
        img_path = os.path.join(video_folder, video_name,str(id).zfill(5))
        img_path += '.jpg'
        l.append(img_path)

    return l



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
    #vidlist = [vid for vid in vidlist if vid.startswith("v_")]
    print "Number of sucessfully donwloaded ",len(vidlist)
    vcount =0
    for videname in reversed(vidlist):
        vcount+=1
        if vcount>0:
            src = vidDir+videname
            numf,width,height,fps,cap = getVidedInfo(src)
            if not cap == -1:

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


if __name__ =='__main__':
    convertVideos()
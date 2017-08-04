import cv2
import os
# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
cropping = False
import pickle

def click_and_crop_fun(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropping

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True
        print 'EVENT_LBUTTONDOWN'

    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        refPt.append((x, y))
        cropping = False
        # draw a rectangle around the region of interest
        cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow("image", image)
        print refPt

def read_pkl(file):
    with open(file,'rb') as f:
        l = pickle.load(f)
        print l
        return l



image_folder = '/home/tcl-admin/data/test'

output_folder = '/home/tcl-admin/data/scense_label'

image_name = os.listdir(image_folder)
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop_fun)
for file in image_name:
    image_path = os.path.join(image_folder,file)
    print 'process ', image_path
    image = cv2.imread(image_path)
    clone = image.copy()


    while True:
        cv2.imshow("image", image)

        key = cv2.waitKey(1) & 0xFF

    # if the 'r' key is pressed, reset the cropping region
        if key == ord("r"):
            print 'reset'
            image = clone.copy()

    # if the 'c' key is pressed, break from the loop
        elif key == ord("c"):
            print 'continue'
            print refPt
            if len(refPt) == 2:
                roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
                roi.fill(255)
                # cv2.imshow("ROI", clone)
                # cv2.waitKey(0)
                out_image_name = 'out_' + file
                image_output = os.path.join(output_folder,out_image_name)
                cv2.imwrite(image_output,clone)
                pkl_out = os.path.join(output_folder,file+'.pkl')
                with open(pkl_out,'wb') as f:
                    pickle.dump(refPt,f)

                assert  read_pkl(pkl_out)==refPt
                refPt = []
            break



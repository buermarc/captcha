import cv2
import sys
import glob
TEST_DATADIR = "data/test/"

breakpoint()
for file in glob.glob(f"{TEST_DATADIR}/*.png"):
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    '''
    crop_img = img[130:170, 40:180]
    cv2.imwrite("test.png", crop_img)
    test_image = cv2.imread("test.png")
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    #add some extra padding around the image
    test_image = cv2.copyMakeBorder(test_image,20,20,20,20, cv2.BORDER_REPLICATE)
    '''
    (thresh, im_bw) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    #find contours 
    contours, hierachy = cv2.findContours(im_bw.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Hack for compatibility with different OpenCV versions
    # contours = contours[0] if imutils.is_cv2() else contours[1]

    letter_image_regions = []
    for contour in contours :
        # get the rectangle of each letter captured
        (x, y, w, h) = cv2.boundingRect(contour)
        im_bw = cv2.rectangle(im_bw,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.imwrite("test-contour.png", im_bw)
    sys.exit(0)

        #compare the width and height of the contour
    '''
    if w / h > 3.0 :
    # it means two letter attached
        half_width = int(w/2)
        letter_image_regions.append((x,y,half_width,h))
        letter_image_regions.append((x+half_width,y,half_width,h))
    else :
        letter_image_regions.append((x,y,w,h))
    '''

    # print(len(letter_image_regions))

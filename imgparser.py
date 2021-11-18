import cv2
import numpy as np
import pytesseract as pytes 
from pytesseract import Output

#Reading image 
img = cv2.imread('DLscan.jpg')

#Custom config to get proper dates
custom_config = r'--oem 3 --psm 12'

# get grayscale image
def get_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# noise removal
def remove_noise(img):
    return cv2.medianBlur(img,5)

class ImagePreProcessor:
    """
    Class for basic image processing functions.
    Takes an image as input.
    Outputs various processed images.
    """
    def __init__(self, image):
        self.image = image

    #thresholding
    def thresholding(self):
        return cv2.threshold(self.image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    #dilation
    def dilate(self):
        kernel = np.ones((5,5),np.uint8)
        return cv2.dilate(self.image, kernel, iterations = 1)
        
    #erosion
    def erode(self):
        kernel = np.ones((5,5),np.uint8)
        return cv2.erode(self.image, kernel, iterations = 1)

    #opening - erosion followed by dilation
    def opening(self):
        kernel = np.ones((5,5),np.uint8)
        return cv2.morphologyEx(self.image, cv2.MORPH_OPEN, kernel)

    #canny edge detection
    def canny(self):
        return cv2.Canny(self.image, 100, 200)

    #skew correction
    def deskew(self):
        coords = np.column_stack(np.where(self.image > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        (h, w) = self.image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(self.image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated

    #template matching
    def match_template(self, template):
        return cv2.matchTemplate(self.image, template, cv2.TM_CCOEFF_NORMED) 




#Function to apply processing 
def ListImage(img):
    """
    Function to return a list of processed images
    Uses the ImagePreProcessor class and its processing methods
    Input: Image
    Output: ImgList
    """
    ImgList = []
    
    gray = get_grayscale(img)
    ImgList.append(gray)

    processor = ImagePreProcessor(gray)
    
    Denoise = remove_noise(img)
    ImgList.append(Denoise)
    
    threshold = processor.thresholding()
    ImgList.append(threshold)
    
    opening = processor.opening()
    ImgList.append(opening)
    
    return ImgList

imageList = ListImage(img)
titleList = ['Grayscale Img', 'Denoised Img', 'thresholded Img', 'Opened Image']

def showImage(titleList,imageList):
    """
    Function to display images from the processed imagelist.
    """
    for title, image in zip(titleList, imageList):

        cv2.imshow(title,image)

        cv2.waitKey(5000)

        return 0

#Denoised image is the best one to grab the date of birth.
#Thresholding seems ok but not crazy

#looping through the list of images to print out the details text
for image in imageList:
    details = pytes.image_to_data(image, output_type=Output.DICT, config=custom_config, lang = 'eng')
    #print(details['text'])
    #print('=========================')


grayed = get_grayscale(img)
process = ImagePreProcessor(grayed)
thresh_img  = process.thresholding()
detials = pytes.image_to_data(thresh_img,output_type=Output.DICT, config=custom_config, lang = 'eng')

#grabbing texts from details dict
total_boxes = len(details['text'])

#looping sequence number from boxes
for sequence_number in range(total_boxes):

    #Outlining the texts recognized from tesseract OCR on the same image
    if int(details['conf'][sequence_number]) >30:

        (x, y, w, h) = (details['left'][sequence_number], details['top'][sequence_number], details['width'][sequence_number],  details['height'][sequence_number])

        threshold_img = cv2.rectangle(thresh_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# display image
# Using cv2.rotate() method
# Using cv2.ROTATE_90_CLOCKWISE rotate
# by 90 degrees clockwise
Limage = cv2.rotate(threshold_img, cv2.ROTATE_90_CLOCKWISE)
Rimage = cv2.rotate(Limage, cv2.ROTATE_90_CLOCKWISE)
Nimage = cv2.rotate(Rimage, cv2.ROTATE_90_CLOCKWISE)


# setting dimensions to resize
scale_percent = 60 # percent of original size
width = int(Nimage.shape[1] * scale_percent / 100) #setting width
height = int(Nimage.shape[0] * scale_percent / 100) #setting height
dim = (width, height) #dimensions
  
# resize image using cv2.resize
resized = cv2.resize(Nimage, dim, interpolation = cv2.INTER_AREA)
cv2.imshow('captured text', resized)

# Maintain output window until user presses a key

cv2.waitKey(0)

# Destroying present windows on screen

cv2.destroyAllWindows()
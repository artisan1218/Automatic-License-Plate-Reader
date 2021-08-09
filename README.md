# Automatic-License-Plate-Reader
An ALPR system based on Python

## Introduction
The ALPR system can be used to detect any license plate number out of an image containing vehicle with a plate.\
Steps:
1. Read in the vehicle image
2. Locate the license plate
3. Extract the license plate
4. Recognize plate number of the extracted plate

## Dataset
This project utilizes the `Car License Plate Detection` dataset from [kaggle](https://www.kaggle.com/andrewmvd/car-plate-detection). This dataset is used to train CNN to locate the license plate. The dataset contains 433 images of cars with plates and corresponding 433 `.xml` file containing the labeled plate location.

## Methodology
To better locate the license plate on a car, CNN is used as neural network can learn more information from different images and therefore is better generalized than pure CV approach.

The structure of CNN is shown below. VGG16 is used here for image recognition
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
vgg16 (Functional)           (None, 6, 6, 512)         14714688  
_________________________________________________________________
dropout (Dropout)            (None, 6, 6, 512)         0         
_________________________________________________________________
flatten (Flatten)            (None, 18432)             0         
_________________________________________________________________
dense (Dense)                (None, 128)               2359424   
_________________________________________________________________
dense_1 (Dense)              (None, 128)               16512     
_________________________________________________________________
dense_2 (Dense)              (None, 64)                8256      
_________________________________________________________________
dense_3 (Dense)              (None, 4)                 260       
=================================================================
```

After locating the plate, easyOCR is used to extract text information out of the license plate. easyOCR has simple syntax and high accuracy for text extraction. One can also use Tesseract to do OCR. 


## Preprocessing
To prepare the dataset for CNN training. All images will be first resized to dimension `(200, 200, 3)` to reduce the computational requirement. For the training label, `beautifulSoup` is used to parse `.xml` file and extract plate location information. Each location is a 4-tuple that contains `xMax, xMin, yMax, yMin` in fixed order. These four coordinates form the top-left and bottom-right corner of the plate, which forms a rectangle that marks the location of plate. 

## Training
The layer structure of CNN is shown below. Note that there is a dropout layer right after VGG16 to pervent overfitting. The output layer has exact four outputs that it can be used to locate a plate. The format is same as the format in `.xml` file.
```
cnn.add(VGG16(weights="imagenet", include_top=False, input_shape=(IMAGE_RESIZE_X, IMAGE_RESIZE_Y, 3)))
cnn.add(keras.layers.Dropout(0.1))
cnn.add(Flatten())
cnn.add(Dense(128, activation="relu"))
cnn.add(Dense(128, activation="relu"))
cnn.add(Dense(64, activation="relu"))
cnn.add(Dense(4, activation="sigmoid"))
```

## CNN Output
The CNN can get ~80% accuracy on the validation set, which is not very accurate. The output result often only covers part of the plate like the one shown below:

![image](https://user-images.githubusercontent.com/25105806/128644180-2f33e36f-6bec-44ab-a1ba-0127874c9a12.png)

To accommodate this issue, the bounding box of the plate(green rectangle) will be enlarged in inversely proportional to the plate-image ratio.
For example, if a plate is small in the original image, we only need to enlarge it by a little bit so that it can cover all plate. However, if a plate is large in the image, which means the image is almost all about this plate, we then need to enlarge it by a lot because the plate itself is already very large compare to the whole image.

Enlarged bounding box that covers the whole plate:\
![3c4cb765624a44cb69bcd35c578fa67](https://user-images.githubusercontent.com/25105806/128785525-0eb7e4f1-5d36-4e11-b24a-82ae1a71b9b8.png)


Consider this big plate, if the bounding box cannot fully cover the whole plate, we then need to make it larger by a lot otherwise the bounding box still cannot cover whole plate.\
<img src="https://user-images.githubusercontent.com/25105806/128644302-9985ccc9-c4c0-4856-8da5-039c2e155754.png" width="50%" height="50%">

However, if there is small plate like this, we only need to enlarge it a little bit and it can cover the whole plate. Otherwise we just make the candidate plate area too large to OCR

<img src="https://user-images.githubusercontent.com/25105806/128644346-2559b3b2-3341-4746-bdc2-cf5145c4243a.png" width="50%" height="50%">


The formula for calculating how should we enlarge the bounding box is shown below, this is only one solution.
```
height_coef = 1 + ((1 / (np.log(box_image_ratio_height))**2) / 2)
width_coef = 1 + ((1 / (np.log(box_image_ratio_width))**2) / 2)
```

## OCR
Next step is to restore the original size of the plate since we normalized all input in CNN to better train the model. Then simply use easyOCR or Tesseract to read the text in detected license plate

Result:
![image](https://user-images.githubusercontent.com/25105806/128644438-5b02378e-f29e-41d3-ba9c-e3dd0fc40ce2.png)


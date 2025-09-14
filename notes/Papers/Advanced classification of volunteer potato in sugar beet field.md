## Advanced classification of volunteer potato in sugar beet field

#### Link to paper
[Nextcloud link](https://nextcloud.sdu.dk/index.php/apps/files/files/295206282?dir=/Shared/Data/Papers&editing=false&openfile=true)

### Chapter 1

#### Requirements
- __The required success rate of volunteer potato control is 95%__
- Ensure less than 5% of undesired control of sugar beet plants
- Ensure a classification time of less than 1 second per field image for real-time processing
- __To ensure the control accuracy of 95%, the classification accuracy has to be higher than that__

#### Fundamental pipeline for weed control

Three core fundamentals:
1. Autonomous vehicle navigation
2. __Weed identification and classifinaction__
3. Actuation of weed removal device

More will be discussed about the 2. part. For weed identification and classification:

1. Any plant materials are first segmented in an acquired image, background pixels are removed and foreground is kept
2. Individual plants are identified
3. Each plant object is classified either as sugar beet plant or volunteer potato plant

#### Problem description
There are two chalanges for vision-based applications that are fully exposed to ambient lighting conditions in an agricartural field:
1. Strongly varying natural illumination
2. Shadows under direct sunlight conditions

In a field environment, illumination conditions constantly change depending on the sky and weather conditions, and this change affects colour pixel values of acquired field images. Shadows influence color values of object in an image scene. The use of conventional features such as colour, shape and texture leds to relatively poor classification result. Besides the colour of plants may change depending on the growth stage and nutritional status.

__This paper tried to evaluate different combinations of colour indices and threshold techniques performs best under different field conditions. It also tried transfer learning based on Convolutional Neural Network for segmentation and classification of volunteer potatoes.__

### Chapter 2 - Ground Shadow detection and removal

Illumination conditions constantly change in an agricultural field environment depending on the sky and weather conditions. Variations greatly affects RGB pixel values of acquired field image. In addition, shadows often create extreme illumination, causing substatial intensity differences within a single image scene. These extremes make vegetation segmentation a challange. Shadows need to be detected and preferably removed for better segmentation permormance.

#### Algorithm - Ground shadow detection and removal
This algorithm uses colour space conversion. Colour pixel values in RGB space can be highly influenced by the illumination conditions. Using a different colour space, that uses a colour representation separating colour and illumination, pixel values are less influenced by illumination conditions. The XYZ colour space was chose, because the normalized form of this colour space separates luminance from colour. The XYS system provides a standard way of describing colours and contains all real colours. 

__The whole pipeline for ground shadow (GS) detection is as follows__:
1. Colour space conversion from RGB to XYZ
2. Contrast enhancement
3. Otsu multilevel threshold

This ground shadow is then substracted from the image that has been converted to the excess green index and threholded with the Otsu threshold - ExG + Otsu - GC.

__Refer to this paper's page 24 for the whole process and equations__. 

Applying shadow removal improved the performance of vegetation segmentation with an average improvement of 20%, 4.4% and 13.5% in precision, specificity and modified accuracy.

### Chapter 3 - Investigation on combinations of colour indices and thresholding methods
In this paper 40 combinations of eight colour indices and five thresholding methods were evaluated. It was also assesed whether it was better to always use one specific combination or the combination should be adapted to the field conditions.
Adapting the combinations to the given conditions yeilded only slightly better results and does not outweight the investment.

__The segmentation of vegetation can be done in three ways:__
1. Using colour-based indices
2. Learning based models
3. Discrete wavelet transform

Chapter 3 is dedicated to the 1. approach. The segmentation of vegetation using colour indices principally contains two steps:
1. Transformation of the RGB image into a near-binary intensity image
2. Applications of a threshold to convert the near-binary image to a full-binary image

__Several method were used for conversion from RGB into near-binary image:__
- ExG - Excess green index
- CIVE - Colour Index of Vegetation Extraction
- NDI - Normalized Difference Index
- ExGR - Excess Green Minus Excess Red
- VEG - Vegetative index
- COM - Combination of Green
- GA - Greenness Accentuation
- HIT - Hue-Invariante Transform 

All of them have been proposed to enhance the difference between the pixel associated with the vegetation and background pixels. For thresholding a fixed threshold value determined by an empirical analysis has been widely used. However this produces poor output when the image is exposed to varying illumination. 

__Used thresholding methods:__
- Otsu method - variance based 
- Riddler - iterative threshold
- Kapur - max entropy threshold
- Rosin - unimodal threshold
- Kittler - minimum error threshold

__Different combinations of these thresholding methods and colour indices have been studied in this paper. For equations to the colour indices and threholding method equations refer to the paper's page 46-51.__

The top five highest performing combinations:
1. CIVE+Kapur
2. CIVE+Rosin
3. CIVE+Kittler
4. ExGR+Kapur
5. GA+Rosin

The poorest performing combinations were:
1. NDI+Kapur
2. ExG+Kapur
3. NDI+Rosin
4. NDI+Kapur
5. VEG+Kapur

CIVE+Kapur showed the best performance with MA of 0.82 and higher. In general varius thresholds worked better in combination with CIVE.

__Chapter 4 goes into the use of bag of visual words, for now we skipp this.__

### Chapter 5 - Transfer learning for the classification of sugar beet and volunteer potato under field conditions
A mentioned above the project had a goal of 95% accuracy of control of volunteer potaoes. This can be more easily achieved using deep learning than classical methods. Training an entire network was not possible due to lack of data, that is why a set of pretrained netwroks were selected, tunned with new data and then evaluated. All were pretrained on ImageNet dataset. These include:
1. AlexNet
2. VGG-19
3. GoogleNet
4. ResNet-50
5. ResNet-101
6. Inception-v3

These were used to classify sugar beet and volunteer potatoes. Transfer learning proved to be effective and showed robust performance. It proposes convolutional neural networks (ConvNet, CNN and their variations) for plant identification. Two options are available in transfer learing: use of ConvNet as a feature extractor and use of ConvNet as a classifier. This paper used a modified and finetunned AlexNet as a feature extractor. Each of the tried networks were modified to have a binary output - sugar beet or volunteer potato. Plant images were resized and data augmentation applied. Afterwards the classification ability of these networks was tested. AlexNet required the least amount of time to classify, however the highest accuracy was obtained with VGG-19 - 98.7%.  
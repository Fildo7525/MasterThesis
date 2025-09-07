### Colour based detection of volunteer potatoes as weeds in sugar beet fields using machine vision


#### Paper link
1. [Link to NextCloud](https://nextcloud.sdu.dk/index.php/apps/files/files/295205268?dir=/Shared/Data/Papers&editing=false&openfile=true)

#### Keywords
- Image analysis
- Crop/weed classification
- Plant-specific weed control


#### Basic information about the paper
- Need for less labour intensive automatic detection system of volunteer potatoes
- Band sprayers have limited success rate of between 20% to 80% of removed volunteer potatoes
- Detection done in sugar beet fields
- Two colour-based machine vision algorithms proposed
- At plant level up to 97% correctly classified
- At another field only 49% correctly identified
- Colour vision approach chosen due to low cost of hardware
- __Features like shape, color and texture can be used__
- __Colour based detection is less comlex and faster then texture and shape based algorithms__
- __Colour has issues with changing natural light conditions__
- Image covered one beet row and two thirds of the soil area of two adjecent
- Images of 640 x 480 pixels
- Images acquired under sunny and cloudy conditions
- About 25% images had no potato plants
- Images were processed and objects classified

#### Image processing
Three steps:
1. Image pre-processing
2. Pixel classification
3. Plant object classification

#### Image pre-processing
1. __Image distortion correction using nonlinear callibration routine__ 
2. __Green plant material segmented from soil__
  - reducing calculations and classification time
  - excessive green parameter EG was used
  - Check for explenation - [Adaptive detection of potato plants](https://github.com/Fildo7525/MasterThesis/blob/master/notes/Papers/Adaptive%20detection%20of%20potato%20plants.md)

3. Remaining plant pixels were transformed using the EGRBI transformation matrix
  - Check for explenation - [Adaptive detection of potato plants](https://github.com/Fildo7525/MasterThesis/blob/master/notes/Papers/Adaptive%20detection%20of%20potato%20plants.md)

#### Pixel Classification
- Combination of K-Means clustering and Bayes classifier. 
- For clustering of image pixels the EG and RB features were used together with Euclidian distance.
- The plant pixels were clustered using K-means with eight randomly chosen cluster centres.
- Volunteer potatos were identified in the EGRB image and manually labelled.
- Corresponding RGB values of the labelled clusters were input as priori data to the Bayes classifier
- After that, all rgb values (256^3) were input to the Bayes detection function and LUT for colour was created.
- LUT had all rgb values, plus and a boolean value for membership of volunteer potato pixels.
- LUT was generated for all learning images and the whole image was classified based on these LUTs

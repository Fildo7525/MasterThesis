## Multispectral Image Segmentation in Agriculture

[Link to Nextcloud](https://nextcloud.sdu.dk/index.php/apps/files/files/296115588?dir=/Shared/Data/Papers&editing=false&openfile=true)

### Abstract
- Multispectral imagery frequently incorporated into agricultural tasks.
- __This work looked into fusion approaches by comparing EFB and NDVI bands/indices.__
- __It also looked into comparison of classical and DL based approaches.__
- In this study the dataset are subjacted to segmentation using classical and DL based approaches on fused images

### NDVI
- A widely used index that relies on the NIR and R bands
- Noemalized Difference Vegetation index (NDVI)
- Provides a quantitative measure of vegetation greenness and density
- Enables more accurate identification and classification of green crops

$$
I_N = \frac{NIR - RED}{NIR + RED}
$$

### Related work
- In the past methods using thresholding, edge based methods and region method have been used.
  - Advantage of simplicity and low computational cost
- Newer approaches using CNNs have been tried
  - Strong advantages in the reduced need of handcrafted featuresV
  - SegNet, Mask-RCNN and so on.

#### Image segmentation
- Involves the task of dividing an image into regions or objects, based on shared characteristics
- __This paper focused on binary segmentation__
  - This means only one class is considered
  - Segmentation mask M = {0,1}  is obtained through a threshold based approach, where T is a threshold value chosen by us, used to distinguish between the positive and negative classes in the segmentation


### Data Fusion
This paper used early and late fusion of RGB and NDVI data for classical and DL based approaches. The Early fusion differes based on the approach.


#### Early Fusion
- Involves combining the information from multiple sources at the input level before the segmentation process.
- Data from multiple sources merged into one representation
- Merging of information in pixel space

##### Early Fusion - Classical Approach
- The RGB image is transformed into a grayscale by using formula:

$$
p_ij^{Gr} = 0.299p_ij^R + 0.587p_ij^G + 0.114p_ij^B
$$

Where p_ij^G, p_ij^R, and p_ij^B are the pixel intensities of the R, G, and B channels at the coordinate (i,j). The resulting grayscale has dimensions of h x w.

- The fused representation is then obtained by comparing the pixel-wise mean between the NDVI and the I_Gr image:

$$
p_ij^{Ec} = \frac{p_ij^{N} + p_ij^{GR}}{2}
$$

Where p_ij^N and p_ij^Gr represent the pixel intensities if the NDVI and the grayscale image

##### Early Fusion - DL based approach

- The resulting fused representation I^Ed  is a tensor of dimensions of h x w x 4.
- It is created using channel wise concatination, concatinating the R, G, B and NDVI bands together

#### Late Fusion
- Done after the seggmentation process has been applied to each individual image.
- The segmentations are obtained and then fused at the later stage 
- Performs the merging at the output space
- Two input images are individually processed through a segmentation model 
- The fused representation is obtained by computing a pixel-wise weighted sum of the class likelihoods from both segmentation models before the final class decision

$$
q_ij^L = \alpha{}q_ij^N + \beta{}q_ij^{RGB}
$$

Where alpha and beta are weights that can be adjusted to balance the contribution of each likelihood according to the models performance.

#### Implementation details
##### Classical method
- Three classical thresholding methods were used for segmentation: 
  1. Otsu
  2. Edge based
  3. Region based 

##### Deep learning
- Two distinct DL-based segmenta
tion models were utilized: SegNet5 and DeepLabV3
-  The training process utilized the AdamW optimizer [9] with a learning rate of 1e-3 for VG
and approximately 1e-4 and 1e-5 for vine models. The Binary Cross-Entropy with Logits Loss (BCEWithLogitsLoss) 

#### Evaluation
-  The classical methods demonstrate to perform well on tasks where the primary objective is to separate foreground from background
- In segmentation tasks that involve identifying spatial regions, where the objective is to detect the plant rows, supervised DL-based approaches show a clear advantage due to their ability to learn spatial information

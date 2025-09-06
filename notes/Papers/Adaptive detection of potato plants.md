### Adaptive detection of volunteer potato plants in sugar beet fields


#### Paper link
1. [Link to NextCloud](https://nextcloud.sdu.dk/index.php/apps/files/files/295200519?dir=/Shared/Data/Papers&editing=false&openfile=true)
2. [Springer paper link](https://link.springer.com/content/pdf/10.1007/s11119-009-9138-9.pdf)

#### Some Interesting Cited papers
Links might include links to sci-hub for normally paid papers, do not share.
1. [Color based detection of volunteer potatoes](https://link.springer.com/content/pdf/10.1007/s11119-007-9044-y.pdf)
2. [Shadow invariant classification for scenes illuminated by daylight](https://sci-hub.se/https://doi.org/10.1364/josaa.17.001952)
3. [Segmentation of row crop plants from weeds using colour and morphology](https://www.sciencedirect.com/science/article/abs/pii/S0168169903000231?via%3Dihub)
4. [Colour based detection of volunteer potatoes as weeds in sugar beet fields using machine vision](https://sci-hub.se/https://doi.org/10.1007/s11119-007-9044-y)
5. [Color indices for weed identification under various soil, residue and lighting conditions](https://sci-hub.se/https://doi.org/10.13031/2013.27838)
6. [Bayesian statistical classifier](https://sci-hub.se/https://doi.org/10.13031/2013.30344)
7. [A survey of image processing techniques for plant
extraction and segmentation in the field](https://sci-hub.se/https://doi.org/10.1016/j.compag.2016.04.024.)
8. [THE ACCURACY OF AUTOMATIC PHOTOGRAMMETRIC TECHNIQUES ON
ULTRA-LIGHT UAV IMAGERY
](https://isprs-archives.copernicus.org/articles/XXXVIII-1-C22/125/2011/isprsarchives-XXXVIII-1-C22-125-2011.pdf)

#### Basic information about the paper
- Volunteer potatos an increasing problem:
- Winter temperaturs not enough to kil tubers
- Poor control results in spread of diseases
- Automatic detection and removal needed
- __Adaptive Bayesian classification method proposed__
- Accuracy in non adaptive schecme was 84.6% (constant light) and 34.9% (changing light conditions)
- Accuracy in adaptive scheme increased to 89.8% (constant light) and 67.7% (changing light conditions)
- __Crop row information was succefully used to train the adaptive classifier, without choosing training data in advance__
- Two colour cameras used - for crop detection and row detection
  - The angles of the cameras and the fields of view were precalbrated before the experiments were started in the field.
- Plant recognition camera images had 50% overlap

#### Classification procedure
For images 1 to N:
1. Determine crop row position in row recognition images
2. Create vegetation grid cells of 100 mm2 in crop recognition images
3. Determine crop row width
4. Determine feature values for each vegetation grid cell
5. Update a priori training data for classification
6. Normalize the feature values
7. Classify each grid cell and show decision

__Vegetation was hihlighted by the excessive green transformation (EG):__
$\\EG = 2G-R-B $
Afterwards histogram approach was used on the overlaid crop row position with the corresponding crop image to get the width of the sugar beat plant. The histogram was made from an image where the bins represent the number of green pizels in the driving direction  in the current image, resulting in the peak at the position of the sugar beet plants, the width of the peak correspond to the width of the plant. Then the crop recognition images were split into cells (about 100mm2). If grid cell contained vegetation using __excessive green transformation__, then six features determined for the specifi grid cell (color information using histograms of the grid cell):
  1. Distance to crop row
  2. Mean red value
  3. Mean green value
  4. mean EG value
  5. Mean red-blue value
  6. Texture in terms of length of the edge segments


__The mean EG and RB values can be calculated is follows:__
$$
\begin{bmatrix}
  EG \\
  RB \\
  I
\end{bmatrix} = 
\begin{bmatrix}
    \frac{-1}{\sqrt{6}} & \frac{2}{\sqrt{6}} & \frac{-1}{\sqrt{6}} \\
    \frac{1}{\sqrt{2}} & 0 & \frac{-1}{\sqrt{2}} \\
    \frac{1}{\sqrt{3}} & \frac{1}{\sqrt{3}} & \frac{1}{\sqrt{3}} \\
\end{bmatrix}
\begin{bmatrix}
    R\\
    G\\
    B
\end{bmatrix}
$$

__The texture measure was calculated as the length of the edge segments within a grid cell.__ The length of the edge segments was calcualted after a Canny edge detection was applied.
__The features 2. to 6. were used in the multivariate Bayesian classification.__ Two classes - volunteer potato and sugar beet. For both classes training features were stored in a buffer of 100 grid cells based on the following function to determine candisdate training cells, where : 

$$
f(a) =
\begin{cases}
a < 1cm \text{= Sugar beet training candidate} \\
1 cm < a < 1.5\sigma cm \text{= to be classified} \\
a > 1.5\sigma cm \text{= Volunteer potato trainig candidate}
\end{cases}
$$
$$ \text{Where } \sigma \text{ is the variance of the crop row width and a represents the distance to the center of the crop row} $$

When both buffers of 100 grid cells full, then the feature values were normalized by substracting the mean and dividing by the standard deviation. Covariancematrix and mean vectors calculated from the buffer. A multivariate Bayesian statistical classifier was used:

$$
d_{j}(x) = ln P(\omega_{j})-\frac{1}{2}ln\left\vert{C_{j}} \right\vert - \frac{1}{2}\left[ (x-m_{j})^T C_{j}^{-1}(x-m_{j})\right]
$$

Then the values of the dj(x) functions for all the grid cells in the image were determined and a grid cell was classified in the class with the highest value for dj(x). The resulting images with classified grid cells were filtered a a lowpass filter to remove all small objects.

For the crop recognition images recorded during the two measurement days, ground truth images were created manually. The grid cells were manually identified as either potatoes ro sugar beets.
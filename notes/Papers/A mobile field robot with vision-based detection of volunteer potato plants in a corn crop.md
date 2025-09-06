###  A mobile field robot with vision-based detection of volunteer potato plants in a corn crop.pdf

#### Paper link
1. [Link to NextCould](https://nextcloud.sdu.dk/index.php/s/3jxYtxQGm335AMx)

#### Some Interesting Cited papers
Links might include links to sci-hub for normally paid papers, do not share.
1. [Imagin spectroscopy for characterisation of grass swards](https://www.researchgate.net/publication/335620092_Imaging_spectroscopy_for_characterisation_of_grass_swards)

#### Basic information about the paper
- Detection of volunteer potatoes from a mobile robot
- Camera at 0.5 m from the ground, aimed perpendicular to the direction of travel (i.e., toward the crop row) and downward
at an angle of 45 degrees to the horizontal. This camera also captures images with a resolution of 320x240 pixels
- Using extended with the DIPImage Scientific Image Processing Toolbox in MATLAB for image processing and PRTools for
statistical analysis
- 6 features for classification:
  - Color:
    1. average red for plant-pixels,
    2. average green for plant-pixels,
    3. average blue for plant-pixels,
  - Size:
    4. total number of plant-pixels in the binary image,
    5. total number of contour pixels in the binary image,
  - Shape:
    6. total number of edge pixels in the binary Canny image.

- The nuimber of edges in potato plant is expected to be higher, due to more cleary defined nerve structure
- Future work refered to wavelet entropy used by Schut (2003) to determine the fractiop of ground covered by clover
in a grass-clover mixture.
- Concluded that is is next to impossible to distinguish plant species in field conditions based on differences in spectral
reflectance.
- **Lab measurement indicates small, but significal difference in reflection at red and infrared bands between leaves of young
potato and corn plands; these differences were too small to have been picked up in the field by the low-cost camera they
used.**

#### Classification procedure
- Features passed into **Fisher linear-discriminant classifier**


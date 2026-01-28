# TODO list

## 27.01.2026

- [x] Create orthomosaic from all the bands and vegetation indices.
- [x] Create create a script to convert png to tif with a tif file reference (georeferenced).
- [x] Export any bands into a png file. This will be the same as the Orthomosaic (the bands are displayed differently in QGIS).

- [ ] **NOT NEEDED** When splitting the orthomosaic `./OpenCV/image_splitter.py`. Write the window coordinates to a CSV file for later reference.
- [ ] In orthomosaic reconstruction, implement a method to handle overlapping areas between image tiles.
- [ ] Check if Henrik's plugins works on the patched up Orthomosaic.

## 28.01.2026

- [x] In orthomosaic reconstruction, implement a method to handle overlapping areas between image tiles.
    - The tiles keep the CRS information so the reference CSV was not needed. The bands and indices are now named correctly.
- [x] Check if Henrik's plugins works on the patched up Orthomosaic.
    - The Orthomosaic can be in uint16 format. However, the referece images need to be in uint8 format. (I used .png)

- [ ] Check other combinations of indices and bands (talk with Samo) for better options. The focus must be on the sharpness.

- Brainstorming:
    - When we use the CDC plugin, we can get the contours of plants
    - Probably some post processing of erode/dilate to get better contours
    - Get the windows out based on the leftover masks
    - Referece it with the step before erosion/dilation to get better masks
    - Use rasterio to get cutouts of the detected plants
    - Gausian blur + Canny edge detection to get better contours
    - Use some potato contours for matching (RANSAC)


## 29.01.2026


# Monitoring of potato crops based on multispectral image feature extraction with vegetation indices

- Multispectral images feature extraction with vegetation indices for potato crops monitoring
- This article discusses the many characteristics that are used to identify plants, such as plant count, \
  plant height estimation, plant area evaluation, plant distance, crop vegetation growth detection, \
  damaged area identification, and higher- lower vegetation
- The Quality Report of Vegetation Indices Value resulted in an accuracy of 96% when good or terrible \
  weather conditions exist

Used indexes:

## NDVI - Normalized Difference Vegetation Index

$$
NDVI = \frac{NIR - R}{NIR + R}
$$

## SVI - Spectral Vegetation Index

$$
SVI = \frac{R - B}{R + B}
$$

## NGRDI - Normalized Red and Green Difference Index

- Used for estimating the biomass of corn and found a linear correlation with C band using python in QGIS

$$
NGRDI = \frac{G - R}{G + R}
$$

## PSRI - Plant Senescence Reflectance Index

- Used for health monitoring stress detecion of platn physiological and crop production and yield analysis

$$
PSRI = \frac{R - B}{NIR}
$$

## GNDVI - Green Normalized Difference Vegetation Index

$$
GNDVI = \frac{NIR - G}{NIR + G}
$$

## RVI - Ratio Vegetation Index

$$
RVI = \frac{NIR}{R}
$$

## NDRE - Normalized Difference Red Edge Index

- cited: "Used for detecting plant status with conditions, NDRE is more sensitive than NDVI"

$$
NDRE = \frac{NIR - RE}{NIR + RE}
$$

## TVI - Transformed Vegetation Index

$$
TVI = 0.5 \times [120 \times (NIR - G) - 200 \times (R - G)]
$$

## CVI - Chlorophyll Vegetation Index

$$
CVI = \frac{NIR \times R}{G^2}
$$

## CIG - Chlorophyll Index Green

$$
CIG = \frac{NIR}{G} - 1
$$

## CIRE - Chlorophyll Index Red Edge

$$
CIRE = \frac{NIR}{RE} - 1
$$

## DVI - Difference Vegetation Index

$$
DVI = NIR \over RE
$$

# Land Surface Temperature

**Would be nice to look up.**

# Worth reading

Panda, S. S., Ames, D. P., & Panigrahi, S. (2010). Application of vegetation indices for agricultural crop yield prediction using neural network techniques.
Remote Sensing, 2(3), 673–96. https://doi.org/10.3390/rs2030673


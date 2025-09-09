# How to use metashape

1. load images (choose separated -> multicamera system)
2. Calibrate reflectance in the top tools -> sun sensor
3. Photos tab -> the dominant image is rht RGB image
4. Change the coordinate system to 32N UTM
 - Repeat for all the datasets
 - They will be in different chunks and save the project in data folder with appropriate name
    - start with date, name, dataset
 - Do not sync with nextcloud

## Batch processing align photos

1. High resolution, other settings should be ok in default.
2. Do it for all the chunks and save after every process.
    1. After done, do verification, first visual
    2. Verification of camera calibration if it makes sense
3. Optimise cameras (star button -> highlight fit additional corrections -> NO adaptive camera matching
4. Save

> !! Do test flights separately, GSD for TF1 is 1cm fro TF2 it is 0.5cm !!

# Build DEM

1. Switch poitcloud to tie points
2. run all
3. set up

## Build ortomosaic



## Pointcloud

1. Duplicatio chunk
2. Extend chunk -> Remove DEM and ortomosaic
and replication with PC -> DEM -> Ortomosaic



- In Model tab
- "0" on the keyboard is gonna reset the view, see if camera is highlighted in the tools
- Align the cameras
  - after align it is not dots
  select all the pictures not alligned -> addjusted estimated values -> sort -> no coordinates are not alligned -> righ-click
  and align the cameras


1. Use classifier instead of object detection to identify only the RoIs from CV

2. Take the ones from overlap. For the bboxes that do not overlap, run the tiles through nn again with lower confidence threshold

3. Use score and weighted combination of the generated confidence
conf = (w_nn * nn_conf + w_cv * cv_conf) ( * iou?)
conf = (w_nn * nn_conf + w_cv * cv_conf) + ( w_iou * iou?)


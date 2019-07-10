# m0
Initial test

# m1
* Batch size 10
* Custom anchor boxes (9 square, between 10 and 90 pixels)
* MAP of 0.95! after 15 epochs
* mini-metro/plugins/SerpentMiniMetroGamePlugin/files/ml_models/station_detector_pt_m1_state_dict.pth

# m2.1
* Modified darknet using single scale feature map, lowest resolution
* Training failed due to bug, later fixed and superceded by m2.3.1

# m2.2
* Modified darknet using single scale feature map, medium resolution
* MAP ~0.9 but very unstable

# m2.3
* Modified darknet using single scale feature map, highest resolution
* MAP ~0.95, inference time on 20 images ~44 seconds

# m2.3.1
* Modified darknet using single scale feature map, highest resolution, with FPN truncated so only using ~40 conv layers instead of ~100
* MAP ~0.95, inference time on 20 images ~28 seconds
* Slow convergence, could try increased LR, or perhaps due to the layer layer being initialised with the weights of completely different layers in the original model (due to the way the darknet weights are loaded), perhaps it would be better to init these with a different distribution. 

# m3
* initial yolo3-tiny test
* using yolo pretrained weights instead of darknet pretrained weights as darknet not available for tiny
* MAP ~0.85, inference time on 20 images ~11 seconds
* Can try using larger image size

# m3.1
* same as m3 but with custom anchors and larger image size (416 -> 608)
* improves MAP to ~0.9 but still not good enough, will likely need a deeper network, somewhere between m3 and m2.3.1
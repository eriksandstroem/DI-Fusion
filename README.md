# DI-Fusion
My baseline version of DI-Fusion for SenFuNet which comprises all datasets used in SenFuNet and the necessary changes to the source code to remove the heuristic outlier filters and remove camera tracking.

Note that I removed the outlier filter which was applied before tracking on the depth maps and also I removed the weight threholding filter. Note that the standard deviation of the estimated TSDF can be thresholded by the marching cubes algorithm. Note also that I added some try-except statements to make the code run without errors.

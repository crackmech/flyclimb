# flyclimb

20190405_FlyGaitPlots.py
20190405_FlyGaitStats.py


20190624_FlyGaitPlots.py

20190624_runAnalysis.py
20190624_stats_plots_final.py

trackStats.py

20190914_20181002_bgSub_ClusterTrack_ImStack.py

20190903_SelectTracksAndClusterTips.py

	
---> fix line 423 in 20190914_20181002_bgSub_ClusterTrack_ImStack.py
---> can we skip the auto clustering part and cluster manually by marking the ROI??
---> clustering done in line# 457 in 20190914_20181002_bgSub_ClusterTrack_ImStack.py


FlowChart for legTip detection and tracking:

---20190914_20181002_bgSub_ClusterTrack_ImStack.py)
1) Select the straight run track
2) Determine the background image from the frames with tracks 
	(Stack all images into a numpy array, select median value for each pixel)
3) Subtract the background frame from each frame
4) Apply binary thresholding (threshold=250)to get the fly from each frame 
	(line#210 20190914_20181002_bgSub_ClusterTrack_ImStack.py)
5) Get fly body center coordinates in each frame
6) Crop the fly for each frame to a 200x200 pixel image based on the body centred frame
7) Apply rat-tail algorithm to get the legs in each frame
8) Erode and dilate (2 times) with a kernel of 5x5 matrix to get body only
9) Apply bitwise XOR for each pixel for the body and thresholded image to get the legs in each frame
10) If legTips are detected in a minimum of 25 frames, cluster the leg tips into 20 clusters using spectral clustering
---> can we skip the auto clustering part and cluster manually by marking the ROI??





Track Stats Pooled for 5 mintues data: 
	20190225_stats_total_final.py
		basefunctions_trackStats.py
			getStatsMultiGrps:
				if pNormality<0.5:
					getRKrusWall
					postHocDunn: fsa.dunnTest
				else:
					getRAnova1
					postHocHSD: agr.HSD_test



Track Stats for per minute data: 
	20190225_stats_perMinute_final.py
	201903029_stats_perMinute.py











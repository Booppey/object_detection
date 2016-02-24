sift: sift.cpp
	g++ -o sift sift.cpp -lopencv_core  -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lopencv_features2d -lopencv_flann -lopencv_nonfree -lopencv_calib3d
clean:
	rm -fr *~ sift

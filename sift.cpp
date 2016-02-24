#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"

using namespace cv;
using namespace std;

int main( int argc, char** argv )
{
    VideoCapture cap(0); //capture the video from webcam

    if ( !cap.isOpened() )  // if not success, exit program
    {
	cout << "Cannot open the web cam" << endl;
	return -1;
    }

    Mat img_object_rgb = imread("images/box.png");
    Mat img_object;
    if( !img_object_rgb.data)
    { std::cout<< " --(!) Error reading images " << std::endl; return -1; }
    cvtColor(img_object_rgb, img_object, CV_RGB2GRAY);

    //-- Step 1: Detect the keypoints using SURF Detector
    int minHessian = 400;
    SurfFeatureDetector detector( minHessian );
    std::vector<KeyPoint> keypoints_object;
    detector.detect( img_object, keypoints_object );

    //-- Step 2: Calculate descriptors (feature vectors)
    SurfDescriptorExtractor extractor;
    Mat descriptors_object;
    extractor.compute( img_object, keypoints_object, descriptors_object );

    while(true)
    {
	Mat img_scene, img_scene_rgb;
        bool b_success = cap.read(img_scene_rgb); // read a new frame from video

	if (!b_success) //if not success, break loop
        {
	    cout << "Cannot read a frame from video stream" << endl;
	    break;
        }
	cvtColor(img_scene_rgb, img_scene, CV_RGB2GRAY);

	std::vector<KeyPoint> keypoints_scene;
	Mat descriptors_scene;
	
	detector.detect( img_scene, keypoints_scene );
	extractor.compute( img_scene, keypoints_scene, descriptors_scene );

	//-- Step 3: Matching descriptor vectors using FLANN matcher
	FlannBasedMatcher matcher;
	std::vector< DMatch > matches;
	matcher.match( descriptors_object, descriptors_scene, matches );

	double max_dist = 0; double min_dist = 100;

	//-- Quick calculation of max and min distances between keypoints
	for( int i = 0; i < descriptors_object.rows; i++ )
	{
	    double dist = matches[i].distance;
	    if( dist < min_dist ) min_dist = dist;
	    if( dist > max_dist ) max_dist = dist;
	}

	std::vector< DMatch > good_matches;
	for( int i = 0; i < descriptors_object.rows; i++ )
	    if( matches[i].distance <= 3*min_dist )  good_matches.push_back( matches[i]);

	Mat img_matches;
	drawMatches( img_object_rgb, keypoints_object, img_scene_rgb, keypoints_scene,
		     good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
		     vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

	//-- Localize the object
	std::vector<Point2f> obj;
	std::vector<Point2f> scene;

	for( int i = 0; i < good_matches.size(); i++ )
	{
	    obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
	    scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
	}

	Mat H = findHomography( obj, scene, CV_RANSAC );

	//-- Get the corners from the object to be "detected"
	std::vector<Point2f> obj_corners(4);
	obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( img_object.cols, 0 );
	obj_corners[2] = cvPoint( img_object.cols, img_object.rows ); obj_corners[3] = cvPoint( 0, img_object.rows );
	std::vector<Point2f> scene_corners(4);

	perspectiveTransform( obj_corners, scene_corners, H);

	//-- Draw lines between the corners (the mapped object in the scene - image_2 )
	line( img_matches, scene_corners[0] + Point2f( img_object.cols, 0), scene_corners[1] + Point2f( img_object.cols, 0), Scalar(0, 255, 0), 4 );
	line( img_matches, scene_corners[1] + Point2f( img_object.cols, 0), scene_corners[2] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
	line( img_matches, scene_corners[2] + Point2f( img_object.cols, 0), scene_corners[3] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
	line( img_matches, scene_corners[3] + Point2f( img_object.cols, 0), scene_corners[0] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );

	//-- Show detected matches
	imshow( "Good Matches & Object detection", img_matches );

	if (waitKey(30) == 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
	{
            cout << "esc key is pressed by user" << endl;
            break;
	}
    }

    return 0;
}

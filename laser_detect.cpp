#include <librealsense/rs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <iostream>
#include <cstdio>
#include <sstream>
#include <chrono>
#include <vector>
#include <algorithm>
#include <math.h>

using namespace cv;
using namespace std;
using namespace rs;


/////////////////////////////////////////////////////////////////////////////
// Create the depth and RGB windows, set their mouse callbacks.
/////////////////////////////////////////////////////////////////////////////

int const INPUT_WIDTH      = 320;
int const INPUT_HEIGHT     = 240;
int const FRAMERATE        = 60;
char* const WINDOW         = "Laser Detection";
bool loop = true;

Scalar txtcolor(0,0,0);
Mat channel[3];
Point orient; Point center;
Point maxLoc; Point minLoc; double minVal; double maxVal; double Vmax;

static void onMouse( int event, int x, int y, int, void* window_name )
{
       if( event == cv::EVENT_LBUTTONDOWN )
       {
             loop = false;
       }
}

void setup_windows( )
{
       cv::namedWindow( WINDOW, 0 );
       cv::setMouseCallback( WINDOW, onMouse, WINDOW );
}

/*
/////////////////////////////////////////////////////////////////////////////
// Image processing functions.
/////////////////////////////////////////////////////////////////////////////

void get_orientation(const uint8_t color_image[], int width, int height)
{
    Mat pre_rgb(Size(width,height), CV_8UC3, color_image);
    cvtColor(pre_rgb, pre_rgb, cv::COLOR_BGR2RGB);
    GaussianBlur(pre_rgb, pre_rgb, Size(width,height), 3.0);
    Mat green = channel[1];
    minMaxLoc(green, &minVal, &maxVal, &minLoc, &maxLoc);
    orient =  Point(maxLoc.x,maxLoc.y);
}

float calc_distance(uint16_t depth_image[], int width, int height, float scale, intrinsics depth_intrin)
{
     uint16_t depth_value_orient = depth_image[orient.y * width + orient.x];
     uint16_t depth_value_center = depth_image[center.y * width + center.x];

     float depth_in_meters_orient = depth_value_orient * scale;
     float depth_in_meters_center = depth_value_center * scale;

     rs::float2 depth_pixel_orient = {(float)orient.x, (float)orient.y};
     rs::float2 depth_pixel_center = {(float)center.x, (float)center.y};

     rs::float3 depth_orient = depth_intrin.deproject(depth_pixel_orient, depth_in_meters_orient);
     rs::float3 depth_center = depth_intrin.deproject(depth_pixel_center, depth_in_meters_center);

     float dist = sqrt(pow((depth_center.x-depth_orient.x),2.0) + pow((depth_center.y-depth_orient.y),2.0));

     return dist;
}
*/

int set_score(float distance) // Get score from distance between orientation and laser point
{
     int score = 0;
     int i = (int)(distance/3);
     switch (i) {
	case 0 : score = 100; break;
	case 1 : score = 80; break;
	case 2 : score = 50; break;
	case 3 : score = 20; break;
	case 4 : score = 10; break;
    	default : score = 0; break;
     }

     return score;
}

/////////////////////////////////////////////////////////////////////////////
// Main loop
/////////////////////////////////////////////////////////////////////////////

int main() try
{
    // Detect device 
    rs::log_to_console(rs::log_severity::warn);
    rs::context ctx;
    printf("There are %d connected RealSense devices.\n", ctx.get_device_count());
    if (ctx.get_device_count() == 0) throw std::runtime_error("No device detected. Is it plugged in?");
    
    // Get device parameters and Prepare streaming
    rs::device * dev = ctx.get_device(0);
    dev->enable_stream(rs::stream::color, INPUT_WIDTH, INPUT_HEIGHT, rs::format::rgb8, FRAMERATE);
    dev->enable_stream(rs::stream::depth, INPUT_WIDTH, INPUT_HEIGHT, rs::format::z16, FRAMERATE);
    auto intrin = dev->get_stream_intrinsics(rs::stream::rectified_color);

    std::cout << " at " << intrin.width << " x " << intrin.height;
    setup_windows( );
    dev->start();

    // Find Green orientation
    for(int i=0; i<120; i++)
    {
    	dev->wait_for_frames();
    	Mat rgb(Size(intrin.width, intrin.height), CV_8UC3, (void*)dev->get_frame_data(rs::stream::rectified_color), Mat::AUTO_STEP);
	Mat hsv;
	cvtColor(rgb, rgb, cv::COLOR_BGR2RGB);      
	cvtColor(rgb, hsv, cv::COLOR_RGB2HSV); 

	Mat mask(rgb.size(), CV_8UC1);   
	Mat res(rgb.size(), CV_8UC3, Scalar(0,0,0));
	int threshold = hsv.at<Vec3b>(intrin.ppy, intrin.ppx)[2]; 
	inRange(rgb,Scalar(0, 50, 0),Scalar(10, 255, 40), mask);
	rgb.copyTo(res, mask);
	inRange(hsv,Scalar(30, 50, 40),Scalar(90, 150, threshold+20), mask);		
	res.copyTo(res, mask);
	split(res,channel);
	Mat green = channel[1];
	minMaxLoc(green, &minVal, &maxVal, &minLoc, &maxLoc);
	orient =  Point(maxLoc.x,maxLoc.y);
	Vmax = maxVal;
    }

    // Time checking start
    int frames = 0;
    float time = 0, fps = 0;
    auto t0 = std::chrono::high_resolution_clock::now();


    while( loop )
    {
        //check for FPS(Frame Per Second)
        auto t1 = std::chrono::high_resolution_clock::now();
        time += std::chrono::duration<float>(t1-t0).count();
        t0 = t1;
	++frames;
        if(time > 0.5f)
        {
            fps = frames / time;
            frames = 0;
            time = 0;
        }

	// Find Red and Bright Point->Laser
        dev->wait_for_frames();
        Mat rgb(Size(intrin.width, intrin.height), CV_8UC3, (void*)dev->get_frame_data(rs::stream::rectified_color), Mat::AUTO_STEP);
   	Mat hsv;
	cvtColor(rgb, rgb, cv::COLOR_BGR2RGB);      
	cvtColor(rgb, hsv, cv::COLOR_RGB2HSV); 

	split(hsv,channel);
	Mat color = channel[0];
	Mat bright = channel[2];
	minMaxLoc(bright, &minVal, &maxVal, &minLoc, &maxLoc);
	center =  Point(maxLoc.x,maxLoc.y);
	double Vpoint = maxVal;
	unsigned char red = color.at<uchar>(maxLoc.x,maxLoc.y);
	float distance = 0.0;
	int score;

        if ( ((Vmax-10) < Vpoint) & (red>130| red<50)){
		circle(rgb, orient, 2, Scalar(0,255,0), -1);  // Scalar(B,G,R)
		circle(rgb, center, 5, Scalar(0,255,255), 2);
		//distance = calc_distance(depth_image, intrin.width, intrin.height , scale, depth_intrin);
		const uint16_t * depth_image = (const uint16_t *)dev->get_frame_data(rs::stream::depth_aligned_to_rectified_color);
    		uint16_t depth_value_orient = depth_image[orient.y * intrin.width + orient.x];
     		uint16_t depth_value_center = depth_image[center.y * intrin.width + center.x];

     		float depth_in_meters_orient = depth_value_orient*0.1;
     		float depth_in_meters_center = depth_value_center*0.1;

     		rs::float2 depth_pixel_orient = {(float)orient.x, (float)orient.y};
     		rs::float2 depth_pixel_center = {(float)center.x, (float)center.y};

     		rs::float3 depth_orient = intrin.deproject(depth_pixel_orient, depth_in_meters_orient);
     		rs::float3 depth_center = intrin.deproject(depth_pixel_center, depth_in_meters_center);

     		distance = sqrt(pow((depth_center.x-depth_orient.x),2.0) + pow((depth_center.y-depth_orient.y),2.0));
		//distance = depth_center.x-depth_orient.x;
		score = set_score(distance);
		txtcolor = Scalar(0,255,0);
	}
	else { 
		circle(rgb, orient, 3, Scalar(0,255,0), -1);
		//int score = 0;      
		txtcolor = Scalar(0,0,255);
	}
		int dist = (int)distance;
	std::ostringstream ss;
	ss << "FPS(Hz): " << floor(fps) << "  " << "Distance(cm): " << dist ;
	putText(rgb, ss.str(), Point(5,13), FONT_HERSHEY_SIMPLEX, 0.4, txtcolor, 1);
	imshow(WINDOW, rgb);
	cvWaitKey(1);
    }
	
    dev->stop();
    destroyAllWindows( );

    return EXIT_SUCCESS;
}

catch(const rs::error & e)
{
    // Method calls against librealsense objects may throw exceptions of type rs::error
    printf("rs::error was thrown when calling %s(%s):\n", e.get_failed_function().c_str(), e.get_failed_args().c_str());
    printf("    %s\n", e.what());
    return EXIT_FAILURE;
}


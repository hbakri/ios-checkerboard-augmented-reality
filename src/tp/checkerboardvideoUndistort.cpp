#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <cstdio>
#include <cstring>
#include <iostream>
#include <ctime>

using namespace cv;
using namespace std;

// Display the help for the program
void help( const char* programName );

// parse the input command line arguments
bool parseArgs( int argc, char**argv, string &inputFilename, string &calibFile );


/******************************************************************/
/* FUNCTION TO DEVELOP                                              */
/******************************************************************/

/**
 * Load the calibration parameters from a file
 *
 * @param[in] calibFilename The name of the file
 * @param[out] matK the 3x3 calibration matrix
 * @param[out] dist the vector containing the distortion coefficients
 * @return true if the parameters have been successfully loaded 
 */
bool loadCameraParameters( const string &calibFilename, Mat &matK, Mat &dist )
{
    // object that will parse the file
    FileStorage fs;

    // open the file to read the parameters
    const bool opened = fs.open(calibFilename, 0);

    // check if the file has been found/opened
    if (!opened) {
        cerr << "Aborting..." << endl;
        return EXIT_FAILURE;
    }

    // load the camera matrix from the tag "camera_matrix" of the file
    fs["camera_matrix"] >> matK;

    // load the distortion coefficients from the tag "distortion_coefficients" of the file
    fs["distortion_coefficients"] >> dist;

    cout << matK << endl;
    cout << dist << endl;

    return true;
}

int main( int argc, char** argv )
{
    /******************************************************************/
    /* CONSTANTS to use                                               */
    /******************************************************************/

    // the name of the window
    const string WINDOW_NAME = "Image View";

    /******************************************************************/
    /* VARIABLES to use                                               */
    /******************************************************************/

    // it will contains the filename of the image file
    string inputFilename;

    // it will contains the filename of the calibration file
    string calibFilename;

    // Used to load the video and get the frames
    VideoCapture capture;

    // Matrix that will contain the camera matrix with the intrinsic parameters
    Mat matK;

    // Matrix that will contain the distortion coefficients of the camera
    Mat dist;

    // variable used to read the user input
    int mode = 'o';

    /******************************************************************/
    /* READ THE INPUT PARAMETERS - DO NOT MODIFY                      */
    /******************************************************************/

    if( !parseArgs( argc, argv, inputFilename, calibFilename ) )
    {
        cerr << "Aborting..." << endl;
        return EXIT_FAILURE;
    }

    /******************************************************************/
    /* PART TO DEVELOP                                                */
    /******************************************************************/

    // create a window using WINDOW_NAME as name to display the image
    namedWindow(WINDOW_NAME, 1);

    // read the input video with capture (same as before)
    const bool opened = capture.open(inputFilename);

    // check it is really opened
    if (!opened) {
        cerr << "Aborting..." << endl;
        return EXIT_FAILURE;
    }

    // call to loadCameraParameters. we want to read the calibration
    // matrix in matK and the distortion coefficients in dist
    loadCameraParameters(calibFilename, matK, dist);

    // processing loop
    while( true )
    {
        Mat view;

        // get the new frame from capture and copy it to view
        capture >> view;

        // if no more images to process exit the loop
        if( view.empty( ) )
            break;

        // this string contains the message to print on the image
        string msg;

        // if we want to see the difference image
        if( mode == 'd' )
        {
            msg = "(o)riginal, (u)ndistorted";

            // temporary image
            Mat temp;

            // copy the original image into temp
            temp = view.clone();

            // apply the undistortion and store the new image in view
            undistort(temp, view, matK, dist);

            // compute the difference between the two images and store the result in view
            absdiff(view, temp, view);
        }
        // if we want to see the undistorted image
        else if( mode == 'u' )
        {
            msg = "(o)riginal, (d)ifference";
            // temporary image
            Mat temp;

            // copy the original image into temp
            temp = view.clone();

            // apply the undistortion and store the new image in view
            undistort(temp, view, matK, dist);
        }
        else
        {
            msg = "(d)ifference, (u)ndistorted";
        }

        int baseLine = 0;
        Size textSize = getTextSize( msg, 1, 1, 1, &baseLine );
        //        cout << baseLine << endl;
        //        cout << textSize << endl;
        Point textOrigin( view.cols / 2 - textSize.width / 2, view.rows - 2 * baseLine - 10 );
        putText( view, msg, textOrigin, 1, 1, Scalar( 0, 255, 0 ) );

        // show view inside the window
        imshow(WINDOW_NAME, view);

        // wait 20ms for user input before processing the next frame
        // Any user input will stop the execution
        int key = 0xff & waitKey( 70 );

        if( key == 'q' )
        {
            break;
        }
        else if( key != 255 )
        {
            mode = key;
        }
    }

    // release the video resource
    capture.release( );

    return EXIT_SUCCESS;
}


// Display the help for the program

void help( const char* programName )
{
    cout << "Undistort the images from a video" << endl
            << "Usage: " << programName << endl
            << "     -c <calib file>                                   # the name of the calibration file" << endl
            << "     <video file>                                      # the name of the video file to process" << endl
            << endl;
}

// parse the input command line arguments

bool parseArgs( int argc, char**argv, string &inputFilename, string &calibFile )
{
    // check the minimum number of arguments
    if( argc < 2 )
    {
        help( argv[0] );
        return false;
    }


    // Read the input arguments
    for( int i = 1; i < argc; i++ )
    {
        const char* s = argv[i];

        if( strcmp( s, "-c" ) == 0 )
        {
            if( i + 1 < argc )
                calibFile.assign( argv[++i] );
            else
            {
                cerr << "Missing argument for option " << s << endl;
                return false;
            }
        }
        else if( s[0] != '-' )
        {
            inputFilename.assign( s );
        }
        else
        {
            cerr << "Unknown option " << s << endl;
            return false;
        }
    }

    return true;
}

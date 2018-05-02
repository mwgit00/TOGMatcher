// MIT License
//
// Copyright(c) 2018 Mark Whitney
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "stdafx.h"

#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

#include <iostream>
#include <sstream>
#include <iomanip>

#include "TOGMatcher.h"
#include "Knobs.h"


using namespace cv;
using namespace std;


#define SCA_BLACK   (cv::Scalar(0,0,0))
#define SCA_RED     (cv::Scalar(0,0,255))
#define SCA_GREEN   (cv::Scalar(0,255,0))


bool wait_and_check_keys(Knobs& rknobs)
{
    bool result = true;

    int nkey = waitKey(1);
    char ckey = static_cast<char>(nkey);

    // check that a keypress has been returned
    if (nkey >= 0)
    {
        if (ckey == 27)
        {
            // done if ESC has been pressed
            result = false;
        }
        else
        {
            rknobs.handle_keypress(ckey);
        }
    }

    return result;
}


void draw_score(cv::Mat& rsrc, const double qmax)
{
    // format score string for viewer (#.##)
    ostringstream oss;
    oss << fixed << setprecision(2) << qmax;

    // draw black background box
    // then draw text score on top of it
    rectangle(rsrc, { 0,0,40,16 }, SCA_BLACK, -1);
    putText(rsrc, oss.str(), { 0,12 }, FONT_HERSHEY_PLAIN, 1.0, SCA_GREEN, 1);
}


void loop(const int ksize)
{
    const char * stitle = "CVMatcher";
    Knobs theKnobs;

    double qmax;
    Size capture_size;
    Size tmpl_offset;
    Point ptmax;
    
    Mat img;
    Mat img_viewer;
    Mat img_gray;
    Mat img_bgr;
    Mat img_channels[3];
    Mat tmatch;
   
    TOGMatcher tmog;
    
    //tmog.create_template_from_file("data\\circle_w_on_b.png", ksize);
    tmog.create_template_from_file("data\\circle_b_on_w.png", ksize);
    //tmog.create_template_from_file("data\\open_box_w_on_b.png", ksize);
    //tmog.create_template_from_file("data\\bottle_20perc_b_on_w.png", ksize);
    //tmog.create_template_from_file("data\\bottle_20perc_cap_b_on_w.png", ksize);
    //tmog.create_template_from_file("data\\bottle_20perc_top_b_on_w.png", ksize);
    //tmog.create_template_from_file("data\\bottle_20perc_curve_b_on_w.png", ksize);

    Mat tdx;
    tmog.get_template_dx().convertTo(tdx, CV_8S);
    Mat tdy;
    tmog.get_template_dy().convertTo(tdy, CV_8S);
    imshow("Template DX", tdx);
    imshow("Template DY", tdy);

    // use size of mask for offset used with output option
    tmpl_offset = tmog.get_template_mask().size();
    tmpl_offset.width /= 2;
    tmpl_offset.height /= 2;

    // need a 0 as argument
    VideoCapture vcap(0);
    if (!vcap.isOpened())
    {
        std::cout << "Failed to open VideoCapture device!" << std::endl;
        ///////
        return;
        ///////
    }

    // camera is ready so grab a first image to determine its full size
    vcap >> img;
    capture_size = img.size();

    // use dummy operation to print initial Knobs help message
    theKnobs.handle_keypress('0');

    // and the image processing loop is running...
    bool is_running = true;

    while (is_running)
    {
        // grab image
        vcap >> img;

        // apply current image scale setting
        double img_scale = theKnobs.get_img_scale();
        Size viewer_size = Size(
            static_cast<int>(capture_size.width * img_scale),
            static_cast<int>(capture_size.height * img_scale));
        resize(img, img_viewer, viewer_size);
        
        // apply current channel setting
        int nchan = theKnobs.get_channel();
        if (nchan == Knobs::ALL_CHANNELS)
        {
            // convert to grayscale
            cvtColor(img_viewer, img_gray, COLOR_BGR2GRAY);
        }
        else
        {
            // select only one BGR channel
            split(img_viewer, img_channels);
            img_gray = img_channels[nchan];
        }
        
        // apply current histogram equalization setting
        if (theKnobs.get_equ_hist_enabled())
        {
            equalizeHist(img_gray, img_gray);
        }

        // apply current pre-processing blur
        int kblur = theKnobs.get_pre_blur();
        if (kblur >= 3)
        {
            GaussianBlur(img_gray, img_gray, { kblur, kblur }, 0, 0);
        }

        // perform template match and locate maximum (best match)
        tmog.perform_match(img_gray, tmatch, theKnobs.get_mask_enabled(), ksize);
        minMaxLoc(tmatch, nullptr, &qmax, nullptr, &ptmax);

        // apply current output mode
        // content varies but all final output images are BGR
        switch (theKnobs.get_output_mode())
        {
            case Knobs::OUT_RAW:
            {
                // show the raw template match result
                // it is shifted and placed on top of blank image of original input size
                Mat full_tmatch = Mat::zeros(img_gray.size(), CV_32F);
                Rect roi = cv::Rect(tmpl_offset.width, tmpl_offset.height, tmatch.cols, tmatch.rows);
                tmatch.copyTo(full_tmatch(roi));
                normalize(full_tmatch, full_tmatch, 0, 1, cv::NORM_MINMAX);
                cvtColor(full_tmatch, img_bgr, COLOR_GRAY2BGR);
                draw_score(img_bgr, qmax);
                imshow(stitle, img_bgr);
                break;
            }
            case Knobs::OUT_MASK:
            {
                // dispaly pre-processed gray input image
                // show red overlay of any matches that exceed arbitrary threshold
                // also show the contour of the best match
                Mat match_mask;
                std::vector<std::vector<cv::Point>> contours;
                cvtColor(img_gray, img_bgr, COLOR_GRAY2BGR);
                normalize(tmatch, tmatch, 0, 1, cv::NORM_MINMAX);
                match_mask = (tmatch > 0.8);
                findContours(match_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
                drawContours(img_bgr, contours, -1, SCA_RED, -1, LINE_8, noArray(), INT_MAX, tmpl_offset);
                drawContours(img_bgr, tmog.get_contours(), -1, SCA_GREEN, 1, LINE_8, noArray(), INT_MAX, ptmax);
                draw_score(img_bgr, qmax);
                imshow(stitle, img_bgr);
                break;
            }
            case Knobs::OUT_COLOR:
            default:
            {
                // show best match on color input image
                drawContours(img_viewer, tmog.get_contours(), -1, SCA_GREEN, 1, LINE_8, noArray(), INT_MAX, ptmax);
                draw_score(img_viewer, qmax);
                imshow(stitle, img_viewer);
                break;
            }
        }

        // handle keyboard events and end when ESC is pressed
        is_running = wait_and_check_keys(theKnobs);
    }

    // when everything is done, release the capture device and windows
    vcap.release();
    destroyAllWindows();
}


int main(int argc, char** argv)
{
    loop(-1);
    return 0;
}

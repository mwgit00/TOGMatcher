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

#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

#include <iostream>
#include <sstream>
#include <iomanip>

#include "TOGMatcher.h"
#include "Knobs.h"
#include "util.h"


using namespace cv;


#define SCA_BLACK   (cv::Scalar(0,0,0))
#define SCA_RED     (cv::Scalar(0,0,255))
#define SCA_GREEN   (cv::Scalar(0,255,0))
#define SCA_YELLOW  (cv::Scalar(0,255,255))


const char * stitle = "TOGMatcher";
int n_record_ctr = 0;


const std::vector<std::string> vfiles =
{
    ".\\data\\circle_b_on_w.png",
    ".\\data\\open_box_w_on_b.png",
    ".\\data\\bottle_20perc_top_b_on_w.png",
    ".\\data\\bottle_20perc_b_on_w.png",
    ".\\data\\bottle_20perc_cap_b_on_w.png",
    ".\\data\\bottle_20perc_curve_b_on_w.png",
};

size_t nfile = 0U;


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


void image_output(
    Mat& rimg,
    const double qmax,
    const Point& rptmax,
    const Knobs& rknobs,
    TOGMatcher& rmatcher)
{
    const Size& roffset = rmatcher.get_template_offset();
    Size ptcenter = { rptmax.x + roffset.width, rptmax.y + roffset.height };

    // format score string for viewer (#.##)
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << qmax;

    // draw black background box
    // then draw text score on top of it
    rectangle(rimg, { 0,0,40,16 }, SCA_BLACK, -1);
    putText(rimg, oss.str(), { 0,12 }, FONT_HERSHEY_PLAIN, 1.0, SCA_GREEN, 1);

    drawContours(rimg, rmatcher.get_contours(), -1, SCA_GREEN, 1, LINE_8, noArray(), INT_MAX, rptmax);
    circle(rimg, ptcenter, 2, SCA_YELLOW, -1);
    if (rknobs.get_record_enabled())
    {
        std::ostringstream osx;
        osx << ".\\movie\\img_" << std::setfill('0') << std::setw(5) << n_record_ctr << ".png";
        imwrite(osx.str(), rimg);
        n_record_ctr++;
        
        // red border around score box if recording
        rectangle(rimg, { 0,0,40,16 }, SCA_RED, 1);
    }
    imshow(stitle, rimg);
}


void reload_template(TOGMatcher& rtmog, const std::string& rs, const int ksize)
{
    const int KPAD = 4;
    const int KW = 240;
    const int KH = 160;
    Mat tdx;
    Mat tdy;
    Mat tdxy = Mat::zeros({ KW, KH }, CV_8S);

    imshow("DX and DY", tdxy);
    
    std::cout << "Loading template:  " << rs << std::endl;
    rtmog.create_template_from_file(rs.c_str(), ksize);

    rtmog.get_template_dx().convertTo(tdx, CV_8S);
    rtmog.get_template_dy().convertTo(tdy, CV_8S);

    // put DX and DY template images in one window
    Size sz = rtmog.get_template_mask().size();
    Rect roix = cv::Rect(KPAD, KPAD, rtmog.get_template_dx().cols, rtmog.get_template_dx().rows);
    Rect roiy = cv::Rect((KW / 2) + KPAD, KPAD, rtmog.get_template_dy().cols, rtmog.get_template_dy().rows);
    tdx.copyTo(tdxy(roix));
    tdy.copyTo(tdxy(roiy));

    imshow("DX and DY", tdxy);
}


void loop(const int ksize)
{
    Knobs theKnobs;
    int op_id;

    double qmax;
    Size capture_size;
    Size tmpl_offset;
    Point ptmax;
    
    Mat img;
    Mat img_viewer;
    Mat img_gray;
    Mat img_channels[3];
    Mat tmatch;
   
    TOGMatcher tmog;

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

    // use dummy operations to print initial Knobs help message
    // and force template to be loaded at start of loop
    theKnobs.handle_keypress('0');
    theKnobs.handle_keypress('t');

    // and the image processing loop is running...
    bool is_running = true;

    while (is_running)
    {
        if (theKnobs.get_op_flag(op_id))
        {
            if (op_id == Knobs::OP_TEMPLATE)
            {
                reload_template(tmog, vfiles[nfile], ksize);
                tmpl_offset = tmog.get_template_offset();
                nfile = (nfile + 1) % vfiles.size();
            }
            else if (op_id == Knobs::OP_RECORD)
            {
                if (theKnobs.get_record_enabled())
                {
                    // reset recording frame counter
                    std::cout << "RECORDING STARTED" << std::endl;
                    n_record_ctr = 0;
                }
                else
                {
                    std::cout << "RECORDING STOPPED" << std::endl;
                }
            }
            else if (op_id == Knobs::OP_MAKE_VIDEO)
            {
                std::cout << "CREATING VIDEO FILE..." << std::endl;
                std::list<std::string> listOfPNG;
                get_dir_list(".\\movie", "*.png", listOfPNG);
                bool is_ok = make_video(15.0, ".\\movie", listOfPNG);
                std::cout << ((is_ok) ? "SUCCESS!" : "FAILURE!") << std::endl;
            }
        }

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
                cvtColor(full_tmatch, img_viewer, COLOR_GRAY2BGR);
                break;
            }
            case Knobs::OUT_MASK:
            {
                // dispaly pre-processed gray input image
                // show red overlay of any matches that exceed arbitrary threshold
                Mat match_mask;
                std::vector<std::vector<cv::Point>> contours;
                cvtColor(img_gray, img_viewer, COLOR_GRAY2BGR);
                normalize(tmatch, tmatch, 0, 1, cv::NORM_MINMAX);
                match_mask = (tmatch > 0.8);
                findContours(match_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
                drawContours(img_viewer, contours, -1, SCA_RED, -1, LINE_8, noArray(), INT_MAX, tmpl_offset);
                break;
            }
            case Knobs::OUT_COLOR:
            default:
            {
                // no extra output processing
                break;
            }
        }

        // always show best match contour and target dot on BGR image
        image_output(img_viewer, qmax, ptmax, theKnobs, tmog);

        // handle keyboard events and end when ESC is pressed
        is_running = wait_and_check_keys(theKnobs);
    }

    // when everything is done, release the capture device and windows
    vcap.release();
    destroyAllWindows();
}


int main(int argc, char** argv)
{
    loop(1);
    return 0;
}

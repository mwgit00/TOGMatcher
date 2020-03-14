// MIT License
//
// Copyright(c) 2020 Mark Whitney
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
#include "opencv2/videoio.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <set>

#include "PatternRec.h"
#include "BGRLandmark.h"
#include "TOGMatcher.h"
#include "Knobs.h"
#include "util.h"


#define MATCH_DISPLAY_THRESHOLD (0.8)           // arbitrary
#define CALIB_PATH              ".\\calib\\"    // user may need to create or change this
#define MOVIE_PATH              ".\\movie\\"    // user may need to create or change this
#define DATA_PATH               ".\\data\\"     // user may need to change this


using namespace cv;


#define SCA_BLACK   (cv::Scalar(0,0,0))
#define SCA_RED     (cv::Scalar(0,0,255))
#define SCA_GREEN   (cv::Scalar(0,255,0))
#define SCA_YELLOW  (cv::Scalar(0,255,255))
#define SCA_BLUE    (cv::Scalar(255,0,0))
#define SCA_WHITE   (cv::Scalar(255,255,255))



enum class max_mode_t : int
{
    NONE = 0,
    RECT,
    CIRCLE,
    CONTOUR,
};

const char * stitle = "TOGMatcher";
int n_record_ctr = 0;


const std::vector<T_file_info> vfiles =
{
    { 0.00, 1.0, "bgrlm9.png"},
    { 0.00, 1.0, "circle_b_on_w.png"},
    { 0.00, 1.0, "bottle_20perc_top_b_on_w.png"},
    { 0.00, 1.0, "bottle_20perc_curve_b_on_w.png"},
    { 0.20, 1.0, "outlet_cover.png"},
    { 0.20, 1.0, "outlet_holes.png" },
    { 0.50, 1.0, "panda_face.png"},
    { 0.00, 1.0, "stars_main.png"}
};

size_t nfile = 0U;



bool check_order(
    const std::vector<int>& rvec,
    const std::map<int, cpoz::BGRLandmark::landmark_info_t>& rmap,
    const int mode)
{
    bool result = true;
    for (size_t ii = 1; ii < rvec.size(); ii++)
    {
        const Point p1 = rmap.at(rvec[ii]).ctr;
        const Point p0 = rmap.at(rvec[ii - 1]).ctr;

        if ((mode == 0) && (p0.x >= p1.x))
        {
            result = false;
            break;
        }

        if ((mode == 1) && (p0.y >= p1.y))
        {
            result = false;
            break;
        }
    }
    return result;
}



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
    const cv::Point& roffset,
    const std::vector<std::vector<cv::Point>>& rcontours,
    const max_mode_t max_mode = max_mode_t::NONE)
{
    Size ptcenter = rptmax + roffset;

    // format score string for viewer (#.##)
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << qmax;

    // draw black background box
    // then draw text score on top of it
    rectangle(rimg, { 0,0,40,16 }, SCA_BLACK, -1);
    putText(rimg, oss.str(), { 0,12 }, FONT_HERSHEY_PLAIN, 1.0, SCA_GREEN, 1);

    switch (max_mode)
    {
        case max_mode_t::RECT:
        {
            Rect roi = Rect(rptmax, rptmax + (roffset * 2));
            rectangle(rimg, roi, SCA_GREEN, 1);
            break;
        }
        case max_mode_t::CIRCLE:
        {
            circle(rimg, ptcenter, 15, SCA_GREEN, 2);
            break;
        }
        case max_mode_t::CONTOUR:
        {
            // draw contours of best match with a yellow dot in the center
            drawContours(rimg, rcontours, -1, SCA_GREEN, 2, LINE_8, noArray(), INT_MAX, rptmax);
            circle(rimg, ptcenter, 2, SCA_YELLOW, -1);
            break;
        }
        case max_mode_t::NONE:
        default:
        {
            break;
        }
    }

    if (rknobs.get_cal_enabled())
    {
        cv::Rect box(Point(0, 0), rimg.size());
        rectangle(rimg, box, SCA_YELLOW, 2);
    }
    
    // save each frame to a file if recording
    if (rknobs.get_record_enabled())
    {
        std::ostringstream osx;
        osx << MOVIE_PATH << "img_" << std::setfill('0') << std::setw(5) << n_record_ctr << ".png";
        imwrite(osx.str(), rimg);
        n_record_ctr++;
        
        // red border around score box if recording
        rectangle(rimg, { 0,0,40,16 }, SCA_RED, 1);
    }
    
    imshow(stitle, rimg);
}



void reload_template(TOGMatcher& rtogm, const T_file_info& rinfo, const int ksize)
{
    const char * sxymtitle = "DX, DY, and Mask";
    const int KPAD = 4;
    const int KW = 480;
    const int KH = 160;
    Mat tdx;
    Mat tdy;
    Mat tmask;
    Mat tdxdym = Mat::zeros({ KW, KH }, CV_8S);
    std::string spath = DATA_PATH + rinfo.sname;

    // clear the window
    imshow(sxymtitle, tdxdym);
    
    std::cout << "Loading template (size= " << ksize << "): " << rinfo.sname << std::endl;
    rtogm.create_template_from_file(spath.c_str(), ksize, rinfo.mag_thr);

    // convert copies of template images into formats suitable for display
    rtogm.get_template_dx().convertTo(tdx, CV_8S);
    rtogm.get_template_dy().convertTo(tdy, CV_8S);
    rtogm.get_template_mask().convertTo(tmask, CV_8S);
    normalize(tmask, tmask, -127, 127, NORM_MINMAX);

    // put DX and DY and mask template images side-by-side in one image
    int ncols = rtogm.get_template_dx().cols;
    int nrows = rtogm.get_template_dx().rows;
    Size sz = rtogm.get_template_mask().size();
    Rect roix = cv::Rect(KPAD, KPAD, ncols, nrows);
    Rect roiy = cv::Rect((KW / 3) + KPAD, KPAD, ncols, nrows);
    Rect roim = cv::Rect(((2 * KW) / 3) + KPAD, KPAD, ncols, nrows);
    tdx.copyTo(tdxdym(roix));
    tdy.copyTo(tdxdym(roiy));
    tmask.copyTo(tdxdym(roim));

    // display the template images
    imshow(sxymtitle, tdxdym);
}



void loop_color_detect(void)
{
    Knobs theKnobs;
    int op_id;

    double qmax = 0.0;
    Size capture_size;
    Point ptmax;

    Mat img;
    Mat img_viewer;
    Mat img_conv;

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

    // and the image processing loop is running...
    bool is_running = true;

    while (is_running)
    {
        // grab image
        vcap >> img;

        // apply the current image scale setting
        double img_scale = 0.5;
        Size viewer_size = Size(
            static_cast<int>(capture_size.width * img_scale),
            static_cast<int>(capture_size.height * img_scale));
        resize(img, img_viewer, viewer_size);

        Mat all = Mat(capture_size, CV_8UC3);

#if 1
        // HSV -> (0-179, 0-255, 0-255)
        // HSV neon pink -> B(H) 160-179, G(S) 120-190, R(V) don't care
        cvtColor(img_viewer, img_conv, COLOR_BGR2HSV);
        std::vector<uint8_t> vlo = { 160, 120, 0 };
        std::vector<uint8_t> vhi = { 179, 190, 255 };
#else
        // YUV neon pink -> B(Y) don't care, G(U) 120-160, R(V) 165-215
        cvtColor(img_viewer, img_conv, COLOR_BGR2YUV);
        std::vector<uint8_t> vlo = { 0, 120, 165 };
        std::vector<uint8_t> vhi = { 255, 160, 215 };
#endif

        // apply the current output mode
        // content varies but all final output images are BGR
        max_mode_t max_mode = max_mode_t::NONE;
        switch (theKnobs.get_output_mode())
        {
        case Knobs::OUT_AUX:
        case Knobs::OUT_RAW:
        case Knobs::OUT_MASK:
        {
            Mat img_thr;
            Mat conv_chan[3];
            cv::split(img_conv, conv_chan);
            max_mode = max_mode_t::RECT;

            // thresholding to match color
            inRange(img_conv, vlo, vhi, img_thr);
            qmax = countNonZero(img_thr);

            Mat aa, bb;
            Rect roi00 = Rect(0, 0, viewer_size.width, viewer_size.height);
            Rect roi01 = Rect(viewer_size.width, 0, viewer_size.width, viewer_size.height);
            Rect roi10 = Rect(0, viewer_size.height, viewer_size.width, viewer_size.height);
            Rect roi11 = Rect(viewer_size.width, viewer_size.height, viewer_size.width, viewer_size.height);

            cvtColor(img_thr, aa, COLOR_GRAY2BGR);
            cvtColor(conv_chan[0], bb, COLOR_GRAY2BGR);
            aa.copyTo(all(roi00));
            img_viewer.copyTo(all(roi01));
            img_conv.copyTo(all(roi10));
            bb.copyTo(all(roi11));
            break;
        }
        case Knobs::OUT_COLOR:
        default:
        {
            // no extra output processing
            max_mode = max_mode_t::NONE;
            break;
        }
        }

        // always show best match contour and target dot on BGR image
        image_output(all, qmax, { 9,9 }, theKnobs, { 3,3 }, {}, max_mode);

        // handle keyboard events and end when ESC is pressed
        is_running = wait_and_check_keys(theKnobs);

        if (theKnobs.get_mask_enabled())
        {
            // hack to dump screenshot and quit if 'm' is pressed
            imwrite("pink_ball.png", img_conv);
            is_running = false;
        }
    }

    // when everything is done, release the capture device and windows
    vcap.release();
    cv::destroyAllWindows();
}



void loop2(void)
{
    const int max_good_ct = 20;

    Knobs theKnobs;
    int op_id;

    std::map<int, cpoz::BGRLandmark::landmark_info_t> cal_label_map;
    std::vector<std::vector<cv::Vec2f>> vvcal;
    std::vector<std::string> vcalfiles;
    int cal_good_ct = 0;
    int cal_ct = 0;
    
    double qmax;
    Size capture_size;
    Point ptmax;

    Mat img;
    Mat img_viewer;
    Mat img_gray;
    Mat tmatch;

    const int kdim = 9;
    const double dthr = 1.6;
	cpoz::BGRLandmark bgrm;
    bgrm.init(kdim, dthr);
	
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

	// use dummy operation to print initial Knobs settings message
	// and force template to be loaded at start of loop
	theKnobs.handle_keypress('0');

    // and the image processing loop is running...
    bool is_running = true;

    while (is_running)
    {
        // grab image
        vcap >> img;

        // apply the current image scale setting
        double img_scale = theKnobs.get_img_scale();
        Size viewer_size = Size(
            static_cast<int>(capture_size.width * img_scale),
            static_cast<int>(capture_size.height * img_scale));
        resize(img, img_viewer, viewer_size);

        // combine all channels into grayscale
        cvtColor(img_viewer, img_gray, COLOR_BGR2GRAY);

        // look for landmarks
        std::vector<cpoz::BGRLandmark::landmark_info_t> qinfo;
        bgrm.perform_match(img_viewer, img_gray, tmatch, qinfo);
        minMaxLoc(tmatch, nullptr, &qmax, nullptr, &ptmax);

#ifdef _COLLECT_SAMPLES
        std::cout << bgrm.samp_ct << std::endl;
#endif

        // apply the current output mode
        // content varies but all final output images are BGR
        max_mode_t max_mode = max_mode_t::NONE;
        switch (theKnobs.get_output_mode())
        {
            case Knobs::OUT_AUX:
            {
                // draw circles around all BGR landmarks and put labels by each one
                // unless about to snap a calibration image which can't have the circles
                // also insert items into map which will also sort them by code
                cal_label_map.clear();
                for (const auto& r : qinfo)
                {
                    cal_label_map[r.code] = r;
                    if ((cal_good_ct < (max_good_ct - 3)) || (cal_good_ct >= max_good_ct))
                    {
                        char x[2] = { 0 };
                        x[0] = static_cast<char>(r.code) + 'A';
                        circle(img_viewer, r.ctr, kdim / 2, (r.diff > 0.0) ? SCA_RED : SCA_BLUE, -1);
                        circle(img_viewer, r.ctr, 2, SCA_WHITE, -1);
                        putText(img_viewer, std::string(x), r.ctr, FONT_HERSHEY_PLAIN, 2.0, SCA_GREEN, 2);
                    }
                }

                // check for proper quantity and uniqueness
                // and do a sanity check to make sure everything lines up properly
                bool is_good_grid = ((qinfo.size() == 12) && (cal_label_map.size() == 12));
                if (is_good_grid)
                {
                    bool y1 = check_order({ 0,1,2 }, cal_label_map, 1);
                    bool y2 = check_order({ 3,4,5 }, cal_label_map, 1);
                    bool y3 = check_order({ 6,7,8 }, cal_label_map, 1);
                    bool y4 = check_order({ 9,10,11 }, cal_label_map, 1);
                    bool x1 = check_order({ 0,3,6,9 }, cal_label_map, 0);
                    bool x2 = check_order({ 1,4,7,10 }, cal_label_map, 0);
                    bool x3 = check_order({ 2,5,8,11 }, cal_label_map, 0);
                    is_good_grid = y1 && y2 && y3 && y4 && x1 && x2 && x3;
                }

                // when calibration mode is enabled
                // the image is dumped to file if pattern passes all the checks
                // lines connecting landmarks go away after image is saved
                // then user must "hide" some landmarks to trigger another grab
                if (theKnobs.get_cal_enabled())
                {
                    if (is_good_grid)
                    {
                        if (cal_good_ct < max_good_ct)
                        {
                            cal_good_ct++;
                            if (cal_good_ct == max_good_ct)
                            {
                                // save calib. image file
                                std::ostringstream osx;
                                osx << "img_" << std::setfill('0') << std::setw(4) << cal_ct << ".png";
                                std::string sfile = osx.str();
                                imwrite(CALIB_PATH + sfile, img_viewer);
                                vcalfiles.push_back(sfile);
                                std::cout << "CALIB. SNAP " << sfile << std::endl;

                                // store the landmark locations
                                std::vector<cv::Vec2f> vimgpts;
                                for (const auto& r : cal_label_map)
                                {
                                    vimgpts.push_back(cv::Vec2f(
                                        static_cast<float>(r.second.ctr.x),
                                        static_cast<float>(r.second.ctr.y)));
                                }
                                vvcal.push_back(vimgpts);
                                cal_ct++;
                            }
                            else
                            {
                                // not saving image so draw lines connecting corners
                                cv::Point prev(0, 0);
                                for (const auto& r : cal_label_map)
                                {
                                    cv::Point pt(r.second.ctr);
                                    if (prev != cv::Point(0, 0))
                                    {
                                        cv::line(img_viewer, prev, pt, SCA_YELLOW, 1);
                                    }
                                    prev = pt;
                                }
                            }
                        }
                    }
                    else
                    {
                        // missed grid detection so start over with countdown
                        cal_good_ct = 0;
                    }
                }
                else
                {
                    // calibration mode turned off
                    cal_good_ct = 0;
                    cal_ct = 0;
                }
                break;
            }
            case Knobs::OUT_RAW:
            {
                // show the raw template match result
                // it is shifted and placed on top of blank image of original input size
                const Point& tmpl_offset = bgrm.get_template_offset();
                Mat full_tmatch = Mat::zeros(img_viewer.size(), CV_32F);
                Rect roi = Rect(tmpl_offset, tmatch.size());
                normalize(tmatch, tmatch, 0, 1, cv::NORM_MINMAX);
                tmatch.copyTo(full_tmatch(roi));
                cvtColor(full_tmatch, img_viewer, COLOR_GRAY2BGR);
                max_mode = max_mode_t::RECT;
                break;
            }
            case Knobs::OUT_MASK:
            {
                // display pre-processed input image
                // show red overlay of any matches that exceed arbitrary threshold
                Mat match_mask;
                std::vector<std::vector<cv::Point>> contours;
                const Point& tmpl_offset = bgrm.get_template_offset();
                cvtColor(img_gray, img_viewer, COLOR_GRAY2BGR);
                normalize(tmatch, tmatch, 0, 1, cv::NORM_MINMAX);
                match_mask = (tmatch > (dthr / 2.0));
                findContours(match_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
                drawContours(img_viewer, contours, -1, SCA_RED, -1, LINE_8, noArray(), INT_MAX, tmpl_offset);
                max_mode = max_mode_t::RECT;
                break;
            }
            case Knobs::OUT_COLOR:
            default:
            {
                // no extra output processing
                max_mode = max_mode_t::RECT;
                break;
            }
        }

        // always show best match contour and target dot on BGR image
        image_output(img_viewer, qmax, ptmax, theKnobs, bgrm.get_template_offset(), {}, max_mode);

        // handle keyboard events and end when ESC is pressed
        is_running = wait_and_check_keys(theKnobs);
    }

    // when everything is done, release the capture device and windows
    vcap.release();
    cv::destroyAllWindows();

    // dump cal data if still in cal mode
    if (theKnobs.get_cal_enabled())
    {
        // the BGRLandmark calibration pattern has 12 corners A-L in ordering shown below
        // so the grid points array must be initialized in same order
        // A D G J
        // B E H K
        // C F I L
        std::vector<Point3f> vgridpts;
        double grid_square = 2.25;
        Size board_size(4, 3);
        for (int j = 0; j < board_size.width; j++)
        {
            for (int i = 0; i < board_size.height; i++)
            {
                vgridpts.push_back(cv::Point3f(float(j * grid_square), float(i * grid_square), 0));
            }
        }

        cv::FileStorage cvfs;
        std::string spath = CALIB_PATH;
        spath += "cal_meta.yaml";
        cvfs.open(spath, cv::FileStorage::WRITE);
        cvfs << "image_size" << capture_size;
        cvfs << "grid_size" << board_size;
        cvfs << "grid_square" << grid_square;
        cvfs << "grid_pts" << vgridpts;
        cvfs << "files" << vcalfiles;
        cvfs << "points" << vvcal;
        cvfs.release();
    }
}



void loop(void)
{
    Knobs theKnobs;
    int op_id;

    double qmax;
    Size capture_size;
    Point ptmax;
    
    Mat img;
    Mat img_viewer;
    Mat img_gray;
    Mat img_channels[3];
    Mat tmatch;
   
    TOGMatcher togm;
    Ptr<CLAHE> pCLAHE = createCLAHE();

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

    // use dummy operation to print initial Knobs settings message
    // and force template to be loaded at start of loop
    theKnobs.handle_keypress('0');

    // initialize template
    reload_template(togm, vfiles[nfile], theKnobs.get_ksize());

    // and the image processing loop is running...
    bool is_running = true;

    while (is_running)
    {
        // check for any operations that
        // might halt or reset the image processing loop
        if (theKnobs.get_op_flag(op_id))
        {
            if (op_id == Knobs::OP_TEMPLATE || op_id == Knobs::OP_KSIZE)
            {
                // changing the template or template kernel size requires a reload
                // changing the template will advance the file index
                if (op_id == Knobs::OP_TEMPLATE)
                {
                    nfile = (nfile + 1) % vfiles.size();
                }
                reload_template(togm, vfiles[nfile], theKnobs.get_ksize());
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
                get_dir_list(MOVIE_PATH, "*.png", listOfPNG);
                bool is_ok = make_video(15.0, MOVIE_PATH,
                    "movie.mov",
                    VideoWriter::fourcc('m', 'p', '4', 'v'),
                    listOfPNG);
                std::cout << ((is_ok) ? "SUCCESS!" : "FAILURE!") << std::endl;
            }
        }

        // grab image
        vcap >> img;

        // apply the current image scale setting
        double img_scale = theKnobs.get_img_scale();
        Size viewer_size = Size(
            static_cast<int>(capture_size.width * img_scale),
            static_cast<int>(capture_size.height * img_scale));
        resize(img, img_viewer, viewer_size);
        
        // apply the current channel setting
        int nchan = theKnobs.get_channel();
        if (nchan == Knobs::ALL_CHANNELS)
        {
            // combine all channels into grayscale
            cvtColor(img_viewer, img_gray, COLOR_BGR2GRAY);
        }
        else
        {
            // select only one BGR channel
            split(img_viewer, img_channels);
            img_gray = img_channels[nchan];
        }
        
        // apply the current histogram equalization setting
        if (theKnobs.get_equ_hist_enabled())
        {
            double c = theKnobs.get_clip_limit();
            pCLAHE->setClipLimit(c);
            pCLAHE->apply(img_gray, img_gray);
        }

        // apply the current blur setting
        int kblur = theKnobs.get_pre_blur();
        if (kblur >= 3)
        {
            GaussianBlur(img_gray, img_gray, { kblur, kblur }, 0, 0);
        }

        // perform template match and locate maximum (best match)
        togm.perform_match(img_gray, tmatch, theKnobs.get_mask_enabled(), theKnobs.get_ksize());
        minMaxLoc(tmatch, nullptr, &qmax, nullptr, &ptmax);

        // apply the current output mode
        // content varies but all final output images are BGR
        max_mode_t max_mode = max_mode_t::NONE;
        switch (theKnobs.get_output_mode())
        {
            case Knobs::OUT_AUX:
            {
                max_mode = max_mode_t::RECT;
                break;
            }
            case Knobs::OUT_RAW:
            {
                // show the raw template match result
                // it is shifted and placed on top of blank image of original input size
                Mat full_tmatch = Mat::zeros(img_gray.size(), CV_32F);
                Rect roi = Rect(togm.get_template_offset(), tmatch.size());
                normalize(tmatch, tmatch, 0, 1, cv::NORM_MINMAX);
                tmatch.copyTo(full_tmatch(roi));
                cvtColor(full_tmatch, img_viewer, COLOR_GRAY2BGR);
                max_mode = max_mode_t::RECT;
                break;
            }
            case Knobs::OUT_MASK:
            {
                // display pre-processed gray input image
                // show red overlay of any matches that exceed arbitrary threshold
                Mat match_mask;
                std::vector<std::vector<cv::Point>> contours;
                const Point& tmpl_offset = togm.get_template_offset();
                cvtColor(img_gray, img_viewer, COLOR_GRAY2BGR);
                normalize(tmatch, tmatch, 0, 1, cv::NORM_MINMAX);
                match_mask = (tmatch > MATCH_DISPLAY_THRESHOLD);
                findContours(match_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
                drawContours(img_viewer, contours, -1, SCA_RED, -1, LINE_8, noArray(), INT_MAX, tmpl_offset);
                max_mode = max_mode_t::RECT;
                break;
            }
            case Knobs::OUT_COLOR:
            default:
            {
                max_mode = max_mode_t::CONTOUR;
                break;
            }
        }

        // update display based on options and mode
        image_output(
            img_viewer,
            qmax,
            ptmax,
            theKnobs,
            togm.get_template_offset(),
            togm.get_contours(),
            max_mode);

        // handle keyboard events and end when ESC is pressed
        is_running = wait_and_check_keys(theKnobs);
    }

    // when everything is done, release the capture device and windows
    vcap.release();
    cv::destroyAllWindows();
}



void test_patt_rec()
{
    // some experimental stuff with PCA...
    PatternRec prfoo;

    prfoo.load_samples_from_img("samples_1K_9keep2.png");

    //prfoo.load_samples_from_img("samples_1K_9x9_markup.png");
    //prfoo.load_samples_from_img("samples_1K_9x9_markup2.png");
    ////prfoo.load_samples_from_img("samples_1K_11x11_markup.png");
    ////prfoo.load_samples_from_img("samples_1K_11x11_markup2.png");
    ////prfoo.load_samples_from_img("samples_1K_13x13_markup.png");
    ////prfoo.load_samples_from_img("samples_1K_15x15_markup.png");
    //
    //// a horizontal flip converts negative samples into positive samples
    //prfoo.load_samples_from_img("samples_1K_9x9_markup.png", -1, true);
    //prfoo.load_samples_from_img("samples_1K_9x9_markup2.png", -1, true);
    ////prfoo.load_samples_from_img("samples_1K_11x11_markup.png", -1, true);
    ////prfoo.load_samples_from_img("samples_1K_11x11_markup2.png", -1, true);
    ////prfoo.load_samples_from_img("samples_1K_13x13_markup.png", -1, true);
    ////prfoo.load_samples_from_img("samples_1K_15x15_markup.png", -1, true);
    
    // dump all the samples...
    prfoo.save_samples_to_csv("train_all");
    
    if (false)
    {
        PatternRec::run_csv_to_pca("train_all_p.csv", "train_all_pca.yaml", 0.8);

        cv::PCA mypca;
        PatternRec::load_pca("train_all_pca.yaml", mypca);

        // test PCA project and back-project to get back DCT components
        cv::Mat samp_pca = mypca.project(prfoo.get_p_sample(88/*988*/));
        cv::Mat samp_dct = mypca.backProject(samp_pca);
        cv::Mat samp_mean = mypca.mean;

        // convert components back to image
        // this better look like a checker board corner
        cv::Mat img_test;
        prfoo.get_dct_fv().features_to_pattern(samp_dct, img_test);
        cv::imwrite("dbg_test_pca.png", img_test);

#if 0
        for (size_t ii = 0; ii < 20; ii++)
        {
            cv::Mat samp_p = mypca.project(prfoo.get_p_sample(ii));
            cv::Mat samp_n = mypca.project(prfoo.get_n_sample(ii));
            cv::Mat samp_0 = mypca.project(prfoo.get_0_sample(ii));
            double qp = cv::norm(samp_p, mypca.project(samp_mean));
            double qn = cv::norm(samp_n, mypca.project(samp_mean));
            double q0 = cv::norm(samp_0, mypca.project(samp_mean));
            std::cout << qp << ", " << qn << ", " << q0 << std::endl;
            std::cout << "--\n";
        }
#endif
    }

    if (false)
    {
        cv::Mat img_p;
        cv::Mat mean_p;
        cv::Mat covar_p;

        cv::Mat img_n;
        cv::Mat mean_n;
        cv::Mat covar_n;

        cv::Mat covar_inv_p;
        cv::Mat covar_inv_n;

        PatternRec::read_csv_into_mat("train_all_p.csv", img_p);
        cv::calcCovarMatrix(img_p, covar_p, mean_p, COVAR_ROWS | COVAR_NORMAL);

        PatternRec::read_csv_into_mat("train_all_n.csv", img_n);
        cv::calcCovarMatrix(img_n, covar_n, mean_n, COVAR_ROWS | COVAR_NORMAL);

        cv::invert(covar_p, covar_inv_p, DECOMP_SVD);
        cv::invert(covar_n, covar_inv_n, DECOMP_SVD);

        std::cout << prfoo.get_dct_fv().get_zigzag_pts() << std::endl;

        // create stats file for BGRLandmark matcher
        std::vector<DCTFeature::T_STATS> vstat;
        vstat.push_back({ mean_p, covar_inv_p, 0.075, "p", true });
        vstat.push_back({ mean_n, covar_inv_n, 0.075, "n", true });
        cv::FileStorage cvfs;
        cvfs.open("bgrm_patt_9.yaml", cv::FileStorage::WRITE);
        cvfs << "dct_kdim" << prfoo.get_dct_fv().dim();
        cvfs << "dct_kmincomp" << prfoo.get_dct_fv().imin();
        cvfs << "dct_kmaxcomp" << prfoo.get_dct_fv().imax();
        cvfs << "stats" << "[";
        for (auto& r : vstat)
        {
            cvfs << "{";
            cvfs << "name" << r.name;
            cvfs << "mean" << r.mean;
            cvfs << "invcov" << r.invcov;
            cvfs << "thr" << r.thr;
            cvfs << "}";
        }
        cvfs << "]";
        cvfs.release();

        DCTFeature dct_foo;
        if (dct_foo.load("bgrm_patt_9.yaml")) std::cout << "loaded new DCT thingy" << std::endl;

        for (size_t ii = 0; ii < 20; ii++)
        {
            std::cout << cv::Mahalanobis(prfoo.get_0_sample(ii), mean_p, covar_inv_p) << ", ";
            std::cout << cv::Mahalanobis(prfoo.get_0_sample(ii), mean_n, covar_inv_n) << ",  ";

            std::cout << cv::Mahalanobis(prfoo.get_p_sample(ii), mean_p, covar_inv_p) << ", ";
            std::cout << cv::Mahalanobis(prfoo.get_p_sample(ii), mean_n, covar_inv_n) << ",  ";

            std::cout << cv::Mahalanobis(prfoo.get_n_sample(ii), mean_p, covar_inv_p) << ", ";
            std::cout << cv::Mahalanobis(prfoo.get_n_sample(ii), mean_n, covar_inv_n) << std::endl;

            std::cout << dct_foo.dist(0, prfoo.get_0_sample(ii)) << ", ";
            std::cout << dct_foo.dist(1, prfoo.get_0_sample(ii)) << ",  ";

            std::cout << dct_foo.dist(0, prfoo.get_p_sample(ii)) << ", ";
            std::cout << dct_foo.dist(1, prfoo.get_p_sample(ii)) << ",  ";

            std::cout << dct_foo.dist(0, prfoo.get_n_sample(ii)) << ", ";
            std::cout << dct_foo.dist(1, prfoo.get_n_sample(ii)) << std::endl;
            std::cout << "--\n";

            if (ii == 0)
            {
                cv::Mat ximg;
                prfoo.get_dct_fv().features_to_pattern(prfoo.get_p_sample(ii), ximg);
                imwrite("db_ximg_p.png", ximg);
                prfoo.get_dct_fv().features_to_pattern(prfoo.get_n_sample(ii), ximg);
                imwrite("db_ximg_n.png", ximg);
                prfoo.get_dct_fv().features_to_pattern(prfoo.get_0_sample(ii), ximg);
                imwrite("db_ximg_0.png", ximg);
            }
        }
    }

    std::cout << "done" << std::endl;
}



void dump_bgrlm_patterns()
{
    // dump all patterns
    for (const auto& r : cpoz::BGRLandmark::PATTERN_MAP)
    {
        cv::Mat img1;
        char c = r.first;
        std::string s = "dbg_bgrlm_" + std::string(1, c) + ".png";
        cpoz::BGRLandmark::create_landmark_image(
            img1, 3.0, 0.25, r.second, { 255,255,255 });
        cv::imwrite(s, img1);
    }
    
    // dump a calibration image
    cv::Mat img2;
    cpoz::BGRLandmark::create_multi_landmark_image(
        img2, cpoz::BGRLandmark::CALIB_LABELS, 4, 3, 0.5, 2.25, 0.25, { 192,192,192 });
    cv::imwrite("dbg_multi.png", img2);
    
    // dump a dual landmark image
    cpoz::BGRLandmark::create_multi_landmark_image(img2, "AG", 2, 1, 0.5, 8, 0.0);
    cv::imwrite("dbg_double.png", img2);

    // dump a quad landmark image
    cpoz::BGRLandmark::create_multi_landmark_image(img2, "AGKE", 2, 2, 1.0, 6.0, 0.0);
    cv::imwrite("dbg_quad.png", img2);

    // dump the gray templates
    cpoz::BGRLandmark bgrm;
    cv::imwrite("dbg_tmpl_gray_p.png", bgrm.get_template_p());
    cv::imwrite("dbg_tmpl_gray_n.png", bgrm.get_template_n());
}



int main(int argc, char** argv)
{
// change 0 to 1 to switch test loops
#if 1
    // test BGRLandmark
    loop2();
#else
    // test TOGMatcher
    // try setting pre-blur kernel size to 5, template kernel size to 5, and enable masking
    loop();
#endif
    return 0;
}

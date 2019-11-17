// MIT License
//
// Copyright(c) 2019 Mark Whitney
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
#define MOVIE_PATH              ".\\movie\\"    // user may need to create or change this
#define DATA_PATH               ".\\data\\"     // user may need to change this


using namespace cv;


#define SCA_BLACK   (cv::Scalar(0,0,0))
#define SCA_RED     (cv::Scalar(0,0,255))
#define SCA_GREEN   (cv::Scalar(0,255,0))
#define SCA_YELLOW  (cv::Scalar(0,255,255))
#define SCA_BLUE    (cv::Scalar(255,0,0))



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
            //circle(rimg, ptcenter, 2, SCA_YELLOW, -1);
            break;
        }
        case max_mode_t::CIRCLE:
        {
            circle(rimg, ptcenter, 15, SCA_GREEN, 2);
            //circle(rimg, ptcenter, 2, SCA_YELLOW, -1);
            break;
        }
        case max_mode_t::CONTOUR:
        {
            // draw contours of best match with a yellow dot in the center
            drawContours(rimg, rcontours, -1, SCA_GREEN, 2, LINE_8, noArray(), INT_MAX, rptmax);
            //circle(rimg, ptcenter, 2, SCA_YELLOW, -1);
            break;
        }
        case max_mode_t::NONE:
        default:
        {
            break;
        }
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



void loop2(void)
{
    const int max_good_ct = 20;

    Knobs theKnobs;
    int op_id;

    cv::FileStorage cvfs;
    std::vector<std::vector<cv::Vec2f>> vvcal;
    std::vector<std::string> vcalfiles;
    std::set<int> cal_label_set;
    int cal_good_ct = 0;
    int cal_ct = 0;
    
    double qmax;
    Size capture_size;
    Point ptmax;

    Mat img;
    Mat img_viewer;
    Mat img_gray;
    Mat tmatch;

	BGRLandmark bgrm;
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

    cvfs.open("cal_meta.yaml", cv::FileStorage::WRITE);

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

        // apply the current histogram equalization setting
        if (theKnobs.get_equ_hist_enabled())
        {
            double c = theKnobs.get_clip_limit();
            pCLAHE->setClipLimit(c);
            pCLAHE->apply(img_gray, img_gray);
        }

        // look for landmarks
        std::vector<BGRLandmark::landmark_info_t> qinfo;
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
                cal_label_set.clear();
                for (const auto& r : qinfo)
                {
                    cal_label_set.insert(r.code);
                    if ((cal_good_ct < (max_good_ct - 3)) || (cal_good_ct >= max_good_ct))
                    {
                        char x[2] = { 0 };
                        x[0] = static_cast<char>(r.code) + 'A';
                        circle(img_viewer, r.ctr, 7, (r.diff > 0.0) ? SCA_RED : SCA_BLUE, 3);
                        putText(img_viewer, std::string(x), r.ctr, FONT_HERSHEY_PLAIN, 2.0, SCA_GREEN, 2);
                    }
                }

                // in this loop, use mask flag as calibration grab mode
                // the image is dumped to file if 12 unique landmarks found (3x4 pattern)
                // then user must "hide" some landmarks to trigger another grab
                if (theKnobs.get_cal_enabled())
                {
                    if (cal_label_set.size() == 12)
                    {
                        if (cal_good_ct < max_good_ct)
                        {
                            cal_good_ct++;
                            if (cal_good_ct == max_good_ct)
                            {
                                // save image file
                                std::ostringstream osx;
                                osx << MOVIE_PATH << "img_" << std::setfill('0') << std::setw(5) << cal_ct << ".png";
                                std::string sfile = osx.str();
                                imwrite(sfile, img_viewer);
                                vcalfiles.push_back(sfile);
                                std::cout << "CALIB. SNAP " << sfile << std::endl;

                                // sort image points by label code
                                std::vector<cv::Vec2f> vimgpts;
                                std::sort(qinfo.begin(), qinfo.end(), BGRLandmark::compare_by_code);
                                for (const auto& r : qinfo)
                                {
                                    vimgpts.push_back(cv::Vec2f(r.ctr.x, r.ctr.y));
                                }
                                vvcal.push_back(vimgpts);
                                cal_ct++;
                            }
                        }
                    }
                    else
                    {
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
                match_mask = (tmatch > 0.9);// MATCH_DISPLAY_THRESHOLD);
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
        // so the corners array must be initialized in same order
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
        cvfs << "image_size" << capture_size;
        cvfs << "grid_size" << board_size;
        cvfs << "grid_square" << grid_square;
        cvfs << "grid_pts" << vgridpts;
        cvfs << "files" << vcalfiles;
        cvfs << "points" << vvcal;
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
    prfoo.load_samples_from_img("samples_1K_9x9_markup.png", 100);
    prfoo.load_samples_from_img("samples_1K_9x9_markup2.png", 100);
    prfoo.load_samples_from_img("samples_1K_11x11_markup.png", 100);
    prfoo.load_samples_from_img("samples_1K_11x11_markup2.png", 100);
    prfoo.load_samples_from_img("samples_1K_13x13_markup.png", 100);
    prfoo.load_samples_from_img("samples_1K_15x15_markup.png", 100);
    
    // due to the nature of the target image, a double flip produces more valid training samples
    prfoo.load_samples_from_img("samples_1K_9x9_markup.png", 100, true);
    prfoo.load_samples_from_img("samples_1K_9x9_markup2.png", 100, true);
    prfoo.load_samples_from_img("samples_1K_11x11_markup.png", 100, true);
    prfoo.load_samples_from_img("samples_1K_11x11_markup2.png", 100, true);
    prfoo.load_samples_from_img("samples_1K_13x13_markup.png", 100, true);
    prfoo.load_samples_from_img("samples_1K_15x15_markup.png", 100, true);
    
    // dump all the samples...
    prfoo.save_samples_to_csv("train_all");
    
    // run PCA with 0.96 variance threshold
    // testing shows this reduces the components down to 9
    // which is symmetrical in the upper left corner of the 8x8 DCT image
    PatternRec::run_csv_to_pca("train_all_p.csv", "train_all_pca.yaml", 0.96);

    cv::PCA mypca;
    PatternRec::load_pca("train_all_pca.yaml", mypca);
    
    // test PCA project and back-project to get back DCT components
    cv::Mat samp_pca = mypca.project(prfoo.get_p_sample(988));
    cv::Mat samp_dct = mypca.backProject(samp_pca);

    // convert components back to image
    // this better look like a checker board corner
    cv::Mat img_test;
    prfoo.get_dct_fv().features_to_pattern(samp_dct, img_test);
    cv::imwrite("dbg_test_pca.png", img_test);
}



int main(int argc, char** argv)
{
    // uncomment line below to test the PCA and DCT stuff and quit
    test_patt_rec(); return 0;

    // uncomment lines below to test reading back cal data
    //std::vector<std::vector<cv::Vec2f>> vvcal;
    //std::vector<std::string> vcalfiles;
    //cv::FileStorage cvfs;
    //cvfs.open("cal_meta.yaml", cv::FileStorage::READ);
    //cvfs["files"] >> vcalfiles;
    //cvfs["points"] >> vvcal;

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

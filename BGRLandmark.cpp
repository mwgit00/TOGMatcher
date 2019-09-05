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

#include <list>
#include "opencv2/highgui.hpp"
#include "BGRLandmark.h"


#define RAIL_MIN(a,b)   {if(a<b){a=b;}}
#define RAIL_MAX(a,b)   {if(a>b){a=b;}}


const cv::Scalar BGRLandmark::BGR_COLORS[8] =
{
    cv::Scalar(0, 0, 0),
    cv::Scalar(0, 0, 255),
    cv::Scalar(0, 255, 0),
    cv::Scalar(0, 255, 255),
    cv::Scalar(255, 0, 0),
    cv::Scalar(255, 0, 255),
    cv::Scalar(255, 255, 0),
    cv::Scalar(255, 255, 255),
};

const cv::Scalar BGRLandmark::BGR_TO_HLS[8] =
{
    cv::Scalar(0, 0, 0),
    cv::Scalar(0, 128, 255),
    cv::Scalar(60, 128, 255),
    cv::Scalar(30, 128, 255),
    cv::Scalar(120, 128, 255),
    cv::Scalar(150, 128, 255),
    cv::Scalar(90, 128, 255),
    cv::Scalar(0, 255, 0),
};

const cv::Scalar BGRLandmark::BGR_TO_HSV[8] =
{
    cv::Scalar(0, 0, 0),
    cv::Scalar(0, 255, 255),
    cv::Scalar(60, 255, 255),
    cv::Scalar(30, 255, 255),
    cv::Scalar(120, 255, 255),
    cv::Scalar(150, 255, 255),
    cv::Scalar(90, 255, 255),
    cv::Scalar(0, 0, 255),
};

const cv::Scalar BGRLandmark::BGR_BORDER = { 128, 128, 128 };

const BGRLandmark::grid_colors_t BGRLandmark::PATTERN_0 = { bgr_t::BLACK, bgr_t::YELLOW, bgr_t::BLACK, bgr_t::CYAN };
const BGRLandmark::grid_colors_t BGRLandmark::PATTERN_1 = { bgr_t::YELLOW, bgr_t::BLACK, bgr_t::CYAN, bgr_t::BLACK };
const BGRLandmark::grid_colors_t BGRLandmark::PATTERN_2 = { bgr_t::BLACK, bgr_t::CYAN, bgr_t::BLACK, bgr_t::YELLOW };
const BGRLandmark::grid_colors_t BGRLandmark::PATTERN_3 = { bgr_t::CYAN, bgr_t::BLACK, bgr_t::YELLOW, bgr_t::BLACK };



BGRLandmark::BGRLandmark()
{
    init(11, PATTERN_0, false);
#if 1
    cv::Mat xx;
    cv::Mat xy;
    create_landmark_image(xx, 3.0, 0.25, PATTERN_0, { 255,255,255 });
    cv::imwrite("foobgrlm.png", xx);
    create_checkerboard_image(xy, 3, 5, 0.5, 0.25);
    cv::imwrite("foobgrcb.png", xy);
#endif
}



BGRLandmark::~BGRLandmark()
{
    // does nothing
}



void BGRLandmark::init(const int k, const grid_colors_t& rcolors, const bool is_rot_45)
{
    // fix k to be odd and in range 3-15
    int fixk = ((k / 2) * 2) + 1;
    RAIL_MIN(fixk, 3);
    RAIL_MAX(fixk, 15);
    
    // stash the color pattern
    pattern = rcolors;
    
    // TODO -- add parameter for threshold
    match_thr = 0.25;

    // create the templates
    cv::Mat tmpl_bgr;
    create_template_image(
        tmpl_bgr,
        fixk,
        { bgr_t::BLACK, bgr_t::WHITE, bgr_t::BLACK, bgr_t::WHITE}, is_rot_45);
    
    cv::cvtColor(tmpl_bgr, tmpl_gray_p, cv::COLOR_BGR2GRAY);
    cv::rotate(tmpl_gray_p, tmpl_gray_n, cv::ROTATE_90_CLOCKWISE);
    imwrite("foogp.png", tmpl_gray_p);
    imwrite("foogn.png", tmpl_gray_n);

    // then create hue template from the BGR template
    cv::Mat img_hls;
    cv::Mat img_channels[3];
    cv::cvtColor(tmpl_bgr, img_hls, cv::COLOR_BGR2HLS);
    split(img_hls, img_channels);
    tmpl_hue = img_channels[0];

    // create a diagonal mask for checking hues
    tmpl_hue_mask = cv::Mat::zeros(tmpl_hue.size(), CV_8UC1);
    const int khalf = tmpl_bgr.size().width / 2;
    //const int hue_sample_ct = 3;
    //const int offset_from_center = 3;
    for (int i = 2; i <= khalf; i++)
    {
        // TODO -- use grid color info
        //img_mask.at<unsigned char>({ koffs - i, koffs - i }) = 255;
        tmpl_hue_mask.at<unsigned char>({ khalf + i, khalf - i }) = 255;
        //img_mask.at<unsigned char>({ koffs + i, koffs + i }) = 255;
        tmpl_hue_mask.at<unsigned char>({ khalf - i, khalf + i }) = 255;
    }

    // TODO -- make these depend on grid dimension somehow
    cv::circle(tmpl_hue_mask, { fixk - 1, 0 }, 3, 255, -1);
    cv::circle(tmpl_hue_mask, { 0, fixk - 1 }, 3, 255, -1);
    //cv::imwrite("hue_mask.png", tmpl_hue_mask);

    // stash offset for this template
    const int fixkh = fixk / 2;
    tmpl_offset.width = fixkh;
    tmpl_offset.height = fixkh;
}



void BGRLandmark::perform_match(
    const cv::Mat& rsrc,
    cv::Mat& rtmatch,
    std::vector<cv::Point>& rpts)
{
    const int xmode = cv::TM_CCOEFF_NORMED;

    cv::Mat tmatch0;
    cv::Mat tmatch1;
    matchTemplate(rsrc, tmpl_gray_p, tmatch0, xmode);
    matchTemplate(rsrc, tmpl_gray_n, tmatch1, xmode);
    rtmatch = (tmatch0 - tmatch1);
    rtmatch = abs(rtmatch);

    // localize each landmark based on absolute threshold
    std::vector<std::vector<cv::Point>> contours;
    cv::Mat match_masked = (rtmatch > 1.7);
    findContours(match_masked, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    for (const auto& r : contours)
    {
        // find centroid associated with each landmark
        // note that contour could be a single point or have area 0
        cv::Moments mm = cv::moments(r, true);
        cv::Point pt = r[0];
        if (mm.m00 > 0.0)
        {
            pt = cv::Point((mm.m10 / mm.m00), (mm.m01 / mm.m00));
        }

        // TODO -- apply hue test
        {
            // all is well so apply template offset and save it
            rpts.push_back({ pt.x + tmpl_offset.width, pt.y + tmpl_offset.height });
        }
    }
}



double BGRLandmark::check_grid_hues(const cv::Mat& rimg, const cv::Point& rpt) const
{
    double result = 0.0;

    // get region of interest around target point
    // match has been done and template offset has already been applied so no need for it here
    const cv::Rect roi = cv::Rect(rpt, tmpl_gray_n.size());
    
    // extract image from region of interest
    // then convert and extract hue channel
    cv::Mat img_hls;
    cv::Mat img_channels[3];
    cv::Mat img_roi(rimg(roi));
    cv::cvtColor(img_roi, img_hls, cv::COLOR_BGR2HLS);
    split(img_hls, img_channels);
    
    //cv::imwrite("sample.png", img_roi);

    // match hues in the non-black-white squares using masked template
    cv::Mat cmatch;
    cv::matchTemplate(img_channels[0], tmpl_hue, cmatch, cv::TM_CCORR_NORMED, tmpl_hue_mask);
    cv::minMaxLoc(cmatch, nullptr, &result, nullptr, nullptr);
    return result;
}



///////////////////////////////////////////////////////////////////////////////
// CLASS STATIC FUNCTIONS

void BGRLandmark::create_template_image(
    cv::Mat& rimg,
    const int k,
    const grid_colors_t& rcolors,
    const bool is_rot_45)
{
    const int kh = k / 2;

    // set colors of each square in 2x2 grid, index is clockwise from upper left
    cv::Scalar colors[4];
    colors[0] = BGR_COLORS[static_cast<int>(rcolors.c00)];
    colors[1] = BGR_COLORS[static_cast<int>(rcolors.c01)];
    colors[2] = BGR_COLORS[static_cast<int>(rcolors.c11)];
    colors[3] = BGR_COLORS[static_cast<int>(rcolors.c10)];

    rimg = cv::Mat::zeros({ k, k }, CV_8UC3);

    // fill in 2x2 blocks (clockwise from upper left)
    cv::rectangle(rimg, { 0, 0, kh, kh }, colors[0], -1);
    cv::rectangle(rimg, { kh + 1, 0, kh, kh }, colors[1], -1);
    cv::rectangle(rimg, { kh, kh, k - 1, k - 1 }, colors[2], -1);
    cv::rectangle(rimg, { 0, kh + 1, kh, kh }, colors[3], -1);

    // fill in average at borders between blocks
    cv::Scalar avg_c00_c10 = (colors[0] + colors[1]) / 2;
    cv::Scalar avg_c10_c11 = (colors[1] + colors[2]) / 2;
    cv::Scalar avg_c11_c01 = (colors[2] + colors[3]) / 2;
    cv::Scalar avg_c01_c00 = (colors[3] + colors[0]) / 2;
    cv::line(rimg, { kh, 0 }, { kh, kh }, avg_c00_c10);
    cv::line(rimg, { kh, kh }, { k - 1, kh }, avg_c10_c11);
    cv::line(rimg, { kh, kh }, { kh, k - 1 }, avg_c11_c01);
    cv::line(rimg, { 0, kh }, { kh, kh }, avg_c01_c00);

    // fill in average of all blocks at central point
    cv::Scalar avg_all = (colors[0] + colors[1] + colors[2] + colors[3]) / 4;
    cv::line(rimg, { kh, kh }, { kh, kh }, avg_all);

    if (is_rot_45)
    {
        // the scale of 1.42 (approx. square root of 2) fills the grid
        const cv::Point2f fctr(static_cast<double>(kh), static_cast<double>(kh));
        cv::Mat rot = cv::getRotationMatrix2D(fctr, 45.0, 1.42);
        cv::warpAffine(rimg, rimg, rot, rimg.size());
    }

    cv::imwrite("foobgr.png", rimg);
}



void BGRLandmark::create_landmark_image(
    cv::Mat& rimg,
    const double dim_grid,
    const double dim_border,
    const grid_colors_t& rcolors,
    const cv::Scalar border_color,
    const bool is_rot_45,
    const int dpi)
{
    // set limits on 2x2 grid size (0.5 inch to 6.0 inch)
    double dim_grid_fix = dim_grid;
    RAIL_MIN(dim_grid_fix, 0.5);
    RAIL_MAX(dim_grid_fix, 6.0);

    // set limits on size of border (0 inches to 1 inch)
    double dim_border_fix = dim_border;
    RAIL_MIN(dim_border_fix, 0.0);
    RAIL_MAX(dim_border_fix, 1.0);

    const int kgrid = static_cast<int>(dim_grid_fix * dpi);
    const int kborder = static_cast<int>(dim_border_fix * dpi);
    const int kgridh = kgrid / 2;
    const int kfull = kgrid + (kborder * 2);

    // set colors of each square in 2x2 grid, index is clockwise from upper left
    cv::Scalar colors[4];
    colors[0] = BGR_COLORS[static_cast<int>(rcolors.c00)];
    colors[1] = BGR_COLORS[static_cast<int>(rcolors.c01)];
    colors[2] = BGR_COLORS[static_cast<int>(rcolors.c11)];
    colors[3] = BGR_COLORS[static_cast<int>(rcolors.c10)];

    // create image that will contain border and grid
    // fill it with border color
    rimg = cv::Mat::zeros({ kfull, kfull }, CV_8UC3);
    cv::rectangle(rimg, { 0, 0, kfull, kfull }, border_color, -1);

    // create image with just the grid
    cv::Mat img_grid = cv::Mat::zeros({ kgrid, kgrid }, CV_8UC3);

    // fill in 2x2 blocks (clockwise from upper left)
    cv::rectangle(img_grid, { 0, 0, kgridh - 1, kgridh - 1 }, colors[0], -1);
    cv::rectangle(img_grid, { kgridh, 0, kgridh, kgridh }, colors[1], -1);
    cv::rectangle(img_grid, { kgridh, kgridh, kgrid - 1, kgrid - 1 }, colors[2], -1);
    cv::rectangle(img_grid, { 0, kgridh, kgridh, kgridh }, colors[3], -1);

    // can rotate 45 degrees to make an "X" pattern instead of a grid
    if (is_rot_45)
    {
        // the half-pixel offset centers things nicely for grid with even dimensions
        // the scale of 1.42 (approx. square root of 2) fills the grid
        const cv::Point2f fctr(static_cast<double>(kgridh) - 0.5, static_cast<double>(kgridh) - 0.5);
        cv::Mat rot = cv::getRotationMatrix2D(fctr, 45.0, 1.42);
        cv::warpAffine(img_grid, img_grid, rot, img_grid.size());
    }

    // copy grid into image with border
    cv::Rect roi = cv::Rect({ kborder, kborder }, img_grid.size());
    img_grid.copyTo(rimg(roi));
}



void BGRLandmark::create_checkerboard_image(
    cv::Mat& rimg,
    const int xrepeat,
    const int yrepeat,
    const double dim_grid,
    const double dim_border,
    const grid_colors_t& rcolors,
    const cv::Scalar border_color,
    const bool is_rot_45,
    const int dpi)
{
    // set limits on 2x2 grid size (0.5 inch to 2.0 inch)
    double dim_grid_fix = dim_grid;
    RAIL_MIN(dim_grid_fix, 0.5);
    RAIL_MAX(dim_grid_fix, 2.0);

    // set limits on size of border (0 inches to 1 inch)
    double dim_border_fix = dim_border;
    RAIL_MIN(dim_border_fix, 0.0);
    RAIL_MAX(dim_border_fix, 1.0);

    const int kgrid = static_cast<int>(dim_grid_fix * dpi);
    const int kborder = static_cast<int>(dim_border_fix * dpi);

    int xrfix;
    int yrfix;

    // set limits on repeat counts
    xrfix = (xrepeat < 2) ? 2 : xrepeat;
    xrfix = (xrepeat > 8) ? 8 : xrepeat;
    yrfix = (yrepeat < 2) ? 2 : yrepeat;
    yrfix = (yrepeat > 8) ? 8 : yrepeat;

    // create a 2x2 grid with no border
    // this will be replicated in the checkerboard
    cv::Mat img_grid;
    create_landmark_image(img_grid, dim_grid_fix, 0.0, rcolors, { 255,255,255 }, is_rot_45, dpi);

    // create image that will contain border and grid
    // fill it with border color
    const int kbx = (kborder * 2) + (img_grid.size().width * xrfix);
    const int kby = (kborder * 2) + (img_grid.size().height * yrfix);
    rimg = cv::Mat::zeros({ kbx, kby }, CV_8UC3);
    cv::rectangle(rimg, { 0, 0, kbx, kby }, border_color, -1);

    // repeat the block pattern
    for (int j = 0; j < yrfix; j++)
    {
        for (int i = 0; i < xrfix; i++)
        {
            cv::Rect roi = { (i * kgrid) + kborder, (j * kgrid) + kborder, kgrid, kgrid };
            img_grid.copyTo(rimg(roi));
        }
    }
}

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

const cv::Scalar BGRLandmark::BGR_BORDER = { 128, 128, 128 };

const BGRLandmark::grid_colors_t BGRLandmark::PATTERN_0 = { bgr_t::BLACK, bgr_t::WHITE, bgr_t::BLACK, bgr_t::WHITE };
const BGRLandmark::grid_colors_t BGRLandmark::PATTERN_A = { bgr_t::BLACK, bgr_t::YELLOW, bgr_t::BLACK, bgr_t::CYAN };
const BGRLandmark::grid_colors_t BGRLandmark::PATTERN_B = { bgr_t::BLACK, bgr_t::YELLOW, bgr_t::BLACK, bgr_t::MAGENTA };
const BGRLandmark::grid_colors_t BGRLandmark::PATTERN_C = { bgr_t::BLACK, bgr_t::CYAN, bgr_t::BLACK, bgr_t::YELLOW };
const BGRLandmark::grid_colors_t BGRLandmark::PATTERN_D = { bgr_t::BLACK, bgr_t::CYAN, bgr_t::BLACK, bgr_t::MAGENTA };
const BGRLandmark::grid_colors_t BGRLandmark::PATTERN_E = { bgr_t::BLACK, bgr_t::MAGENTA, bgr_t::BLACK, bgr_t::CYAN };
const BGRLandmark::grid_colors_t BGRLandmark::PATTERN_F = { bgr_t::BLACK, bgr_t::MAGENTA, bgr_t::BLACK, bgr_t::YELLOW };



BGRLandmark::BGRLandmark()
{
    init();
#if _DEBUG
    cv::Mat img1;
    cv::Mat img2;
    create_landmark_image(img1, 3.0, 0.25, PATTERN_A, { 255,255,255 });
    cv::imwrite("dbg_bgrlm.png", img1);
    create_checkerboard_image(img2, 3, 5, 0.5, 0.25);
    cv::imwrite("dbg_bgrcb.png", img2);
#endif
}



BGRLandmark::~BGRLandmark()
{
    // does nothing
}



void BGRLandmark::init(
    const int k,
    const double thr_corr,
    const int thr_pix_rng,
    const int thr_pix_min)
{
    // fix k to be odd and in range 9-15
    int fixk = ((k / 2) * 2) + 1;
    RAIL_MIN(fixk, 9);
    RAIL_MAX(fixk, 15);
    
    // apply thresholds
    // TODO -- assert type is CV_8U somewhere during match
    this->thr_corr = thr_corr;
    this->thr_pix_rng = thr_pix_rng;
    this->thr_pix_min = thr_pix_min;

    // create the B&W matching templates
    cv::Mat tmpl_bgr;
    create_template_image(tmpl_bgr, fixk, PATTERN_0);
    cv::cvtColor(tmpl_bgr, tmpl_gray_p, cv::COLOR_BGR2GRAY);
    cv::rotate(tmpl_gray_p, tmpl_gray_n, cv::ROTATE_90_CLOCKWISE);

#ifdef _DEBUG
    imwrite("dbg_tmpl_gray_p.png", tmpl_gray_p);
    imwrite("dbg_tmpl_gray_n.png", tmpl_gray_n);
#endif

    // stash offset for this template
    const int fixkh = fixk / 2;
    tmpl_offset.x = fixkh;
    tmpl_offset.y = fixkh;
}



void BGRLandmark::perform_match(
    const cv::Mat& rsrc,
    cv::Mat& rtmatch,
    std::vector<BGRLandmark::landmark_info_t>& rinfo)
{
    const int xmode = cv::TM_CCOEFF_NORMED;

    // match the positive and negative templates
    // and find absolute difference between the two results
    cv::Mat tmatch0;
    cv::Mat tmatch1;
    matchTemplate(rsrc, tmpl_gray_p, tmatch0, xmode);
    matchTemplate(rsrc, tmpl_gray_n, tmatch1, xmode);
    rtmatch = abs(tmatch0 - tmatch1);

    // find local maxima in the match results...
    cv::Mat maxima_mask;
    cv::dilate(rtmatch, maxima_mask, cv::Mat());
    cv::compare(rtmatch, maxima_mask, maxima_mask, cv::CMP_GE);

    // then apply absolute threshold to get the best local maxima
    std::vector<std::vector<cv::Point>> contours;
    cv::Mat match_masked = (rtmatch > thr_corr);
    maxima_mask = maxima_mask & match_masked;

    // collect point locations of all local maxima
    std::vector<cv::Point> vec_maxima_pts;
    cv::findNonZero(maxima_mask, vec_maxima_pts);

    // check each maxima...
    for (const auto& rpt : vec_maxima_pts)
    {
        // positive diff means black in upper-left/lower-right
        // negative diff means black in lower-left/upper-right
        float pix_p = tmatch0.at<float>(rpt);
        float pix_n = tmatch1.at<float>(rpt);
        float diff = pix_p - pix_n;

        // extract region of interest
        const cv::Rect roi = cv::Rect(rpt, tmpl_gray_p.size());
        cv::Mat img_roi(rsrc(roi));

        // get pixel range stats in ROI
        double min_roi;
        double max_roi;
        cv::minMaxLoc(img_roi, &min_roi, &max_roi);
        double rng_roi = max_roi - min_roi;

        // a landmark ROI should have two dark squares and and two light squares
        // see if ROI has large range in pixel values and a minimum that is sufficiently dark
        if ((rng_roi > thr_pix_rng) && (min_roi < thr_pix_min))
        {
            // stuff it
            // TODO -- add some kind of additional shape test
            landmark_info_t lminfo{ rpt + tmpl_offset, diff, rng_roi, min_roi };
            rinfo.push_back(lminfo);
        }
    }
}



int BGRLandmark::identify_colors(const cv::Mat& rimg, const BGRLandmark::landmark_info_t& rinfo) const
{
    int result = -1;

    // get BGR region of interest around target point
    // template offset has added so subtract it again
    const cv::Point ctr_offset = rinfo.ctr - tmpl_offset;
    const cv::Rect roi = cv::Rect(ctr_offset, tmpl_gray_p.size());
    
    // extract image from region of interest
    cv::Mat img_roi(rimg(roi));

    // do median blur with size 1/3 of template size (forced to be odd)
    int k = tmpl_gray_n.size().width / 3;
    if ((k % 2) == 0) { k += 1; }
    cv::Mat img_roi_blur;
    cv::medianBlur(img_roi, img_roi_blur, k);

    // TODO -- identify yellow, cyan, magenta

    return result;
}



///////////////////////////////////////////////////////////////////////////////
// CLASS STATIC FUNCTIONS

void BGRLandmark::create_template_image(
    cv::Mat& rimg,
    const int k,
    const grid_colors_t& rcolors)
{
    const int kh = k / 2;

    // set colors of each square in 2x2 grid, index is clockwise from upper left
    cv::Scalar colors[4];
    colors[0] = BGR_COLORS[static_cast<int>(rcolors.c00)];
    colors[1] = BGR_COLORS[static_cast<int>(rcolors.c01)];
    colors[2] = BGR_COLORS[static_cast<int>(rcolors.c11)];
    colors[3] = BGR_COLORS[static_cast<int>(rcolors.c10)];

    rimg = cv::Mat::zeros({ k, k }, CV_8UC3);

    // fill in 2x2 squares (clockwise from upper left)
    cv::rectangle(rimg, { 0, 0, kh, kh }, colors[0], -1);
    cv::rectangle(rimg, { kh + 1, 0, kh, kh }, colors[1], -1);
    cv::rectangle(rimg, { kh, kh, k - 1, k - 1 }, colors[2], -1);
    cv::rectangle(rimg, { 0, kh + 1, kh, kh }, colors[3], -1);

    // fill in average at borders between squares
    cv::Scalar avg_c00_c10 = (colors[0] + colors[1]) / 2;
    cv::Scalar avg_c10_c11 = (colors[1] + colors[2]) / 2;
    cv::Scalar avg_c11_c01 = (colors[2] + colors[3]) / 2;
    cv::Scalar avg_c01_c00 = (colors[3] + colors[0]) / 2;
    cv::line(rimg, { kh, 0 }, { kh, kh }, avg_c00_c10);
    cv::line(rimg, { kh, kh }, { k - 1, kh }, avg_c10_c11);
    cv::line(rimg, { kh, kh }, { kh, k - 1 }, avg_c11_c01);
    cv::line(rimg, { 0, kh }, { kh, kh }, avg_c01_c00);

    // central point gets average of all squares
    cv::Scalar avg_all = (colors[0] + colors[1] + colors[2] + colors[3]) / 4;
    cv::line(rimg, { kh, kh }, { kh, kh }, avg_all);

#ifdef _DEBUG
    cv::imwrite("dbg_tmpl_bgr.png", rimg);
#endif
}



void BGRLandmark::create_landmark_image(
    cv::Mat& rimg,
    const double dim_grid,
    const double dim_border,
    const grid_colors_t& rcolors,
    const cv::Scalar border_color,
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
    create_landmark_image(img_grid, dim_grid_fix, 0.0, rcolors, { 255,255,255 }, dpi);

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

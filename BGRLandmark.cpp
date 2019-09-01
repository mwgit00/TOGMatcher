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

const cv::Scalar BGRLandmark::BGR_BORDER = { 255, 255, 255 };

const BGRLandmark::grid_colors_t BGRLandmark::PATTERN_0 = { bgr_t::BLACK, bgr_t::YELLOW, bgr_t::BLACK, bgr_t::CYAN };
const BGRLandmark::grid_colors_t BGRLandmark::PATTERN_1 = { bgr_t::YELLOW, bgr_t::BLACK, bgr_t::CYAN, bgr_t::BLACK };
const BGRLandmark::grid_colors_t BGRLandmark::PATTERN_2 = { bgr_t::BLACK, bgr_t::CYAN, bgr_t::BLACK, bgr_t::YELLOW };
const BGRLandmark::grid_colors_t BGRLandmark::PATTERN_3 = { bgr_t::CYAN, bgr_t::BLACK, bgr_t::YELLOW, bgr_t::BLACK };



BGRLandmark::BGRLandmark()
{
    init();
    cv::Mat xx;
    cv::Mat xy;
    create_landmark_image(xx, 3.0, 0.25);
    cv::imwrite("foobgrlm.png", xx);
    augment_landmark_image(xx, 0653); // octal
    cv::imwrite("foobgraug.png", xx);

    create_checkerboard_image(xy, 3, 5, 0.5, 0.25);
    cv::imwrite("foobgrcb.png", xy);
}



BGRLandmark::~BGRLandmark()
{
    // does nothing
}



void BGRLandmark::init(const int k, const grid_colors_t& rcolors, const int mode)
{
    int fixk;
    int fixkh;

    // set limits on size and make it odd
    fixk = (k < 3) ? 3 : k;
    fixk = (k > 15) ? 15 : k;
    fixk = ((fixk % 2) == 0) ? fixk + 1 : fixk;
    fixkh = fixk / 2;
    
    // apply mode for template match
    // TM_CCOEFF seems like best all-around choice
    this->mode = mode;

    match_thr = 0.75;

    // stash the template image
    create_template_image(tmpl_bgr, fixk, rcolors);

    // stash offset for this template
    tmpl_offset.width = fixkh;
    tmpl_offset.height = fixkh;
}



void BGRLandmark::perform_match(
    const cv::Mat& rsrc_bgr,
    cv::Mat& rtmatch,
    std::vector<std::vector<cv::Point>>& rcontours)
{
    matchTemplate(rsrc_bgr, tmpl_bgr, rtmatch, mode);

    double qmax;
    std::vector<std::vector<cv::Point>> contours;

    minMaxLoc(rtmatch, nullptr, &qmax, nullptr, nullptr);
    cv::Mat match_masked = (rtmatch > (match_thr * qmax));
    findContours(match_masked, rcontours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
}


void BGRLandmark::perform_match_cb(
    const cv::Mat& rsrc,
    cv::Mat& rtmatch)
{
    const int k = 7;
    cv::Mat t0;
    cv::Mat t1;
    cv::Mat tmatch0;
    cv::Mat tmatch1;
    grid_colors_t PATTERN_0_INV = invert_grid_colors(PATTERN_0);
    create_template_image(t0, k, PATTERN_0);
    create_template_image(t1, k, PATTERN_0_INV);
    const int xmode = cv::TM_CCOEFF; // TM_CCOEFF, TM_CCORR_NORMED good
    matchTemplate(rsrc, t0, tmatch0, xmode);
    matchTemplate(rsrc, t1, tmatch1, xmode);
    rtmatch = (tmatch0 - tmatch1);
    rtmatch = abs(rtmatch);
}



BGRLandmark::bgr_t BGRLandmark::invert_bgr(const bgr_t color)
{
    bgr_t result = bgr_t::BLACK;
    switch (color)
    { 
        case bgr_t::BLACK: result = bgr_t::WHITE; break;
        case bgr_t::RED: result = bgr_t::CYAN; break;
        case bgr_t::GREEN: result = bgr_t::MAGENTA; break;
        case bgr_t::YELLOW: result = bgr_t::BLUE; break;
        case bgr_t::BLUE: result = bgr_t::YELLOW; break;
        case bgr_t::MAGENTA: result = bgr_t::GREEN; break;
        case bgr_t::CYAN: result = bgr_t::RED; break;
        case bgr_t::WHITE: result = bgr_t::BLACK; break;
        default: break;
    }
    return result;
}



BGRLandmark::grid_colors_t BGRLandmark::invert_grid_colors(const grid_colors_t& rcolors)
{
    grid_colors_t result;
    result.c00 = invert_bgr(rcolors.c00);
    result.c01 = invert_bgr(rcolors.c01);
    result.c11 = invert_bgr(rcolors.c11);
    result.c10 = invert_bgr(rcolors.c10);
    return result;
}



void BGRLandmark::create_template_image(cv::Mat& rimg, int k, const grid_colors_t& rcolors)
{
    int fixk;
    int fixkh;

    // set limits on size
    fixk = (k < 3) ? 3 : k;
    fixk = (k > 15) ? 15 : k;
    fixkh = fixk / 2;

    // set colors of each square in 2x2 grid, index is clockwise from upper left
    cv::Scalar colors[4];
    colors[0] = BGR_COLORS[static_cast<int>(rcolors.c00)];
    colors[1] = BGR_COLORS[static_cast<int>(rcolors.c01)];
    colors[2] = BGR_COLORS[static_cast<int>(rcolors.c11)];
    colors[3] = BGR_COLORS[static_cast<int>(rcolors.c10)];

    rimg = cv::Mat::zeros({ fixk, fixk }, CV_8UC3);

    // fill in 2x2 blocks (clockwise from upper left)
    cv::rectangle(rimg, { 0, 0, fixkh, fixkh }, colors[0], -1);
    cv::rectangle(rimg, { fixkh + 1, 0, fixkh, fixkh }, colors[1], -1);
    cv::rectangle(rimg, { fixkh, fixkh, fixk - 1, fixk - 1 }, colors[2], -1);
    cv::rectangle(rimg, { 0, fixkh + 1, fixkh, fixkh }, colors[3], -1);

    // fill in average at borders between blocks
    cv::Scalar avg_c00_c10 = (colors[0] + colors[1]) / 2;
    cv::Scalar avg_c10_c11 = (colors[1] + colors[2]) / 2;
    cv::Scalar avg_c11_c01 = (colors[2] + colors[3]) / 2;
    cv::Scalar avg_c01_c00 = (colors[3] + colors[0]) / 2;
    cv::line(rimg, { fixkh, 0 }, { fixkh, fixkh }, avg_c00_c10);
    cv::line(rimg, { fixkh, fixkh }, { fixk - 1, fixkh }, avg_c10_c11);
    cv::line(rimg, { fixkh, fixkh }, { fixkh, fixk - 1 }, avg_c11_c01);
    cv::line(rimg, { 0, fixkh }, { fixkh, fixkh }, avg_c01_c00);

    // fill in average of all blocks at central point
    cv::Scalar avg_all = (colors[0] + colors[1] + colors[2] + colors[3]) / 4;
    cv::line(rimg, { fixkh, fixkh }, { fixkh, fixkh }, avg_all);

    cv::imwrite("foobgr.png", rimg);
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

    cv::Rect roi = cv::Rect({ kborder, kborder }, img_grid.size());

    // fill in 2x2 blocks (clockwise from upper left)
    cv::rectangle(img_grid, { 0, 0, kgridh - 1, kgridh - 1 }, colors[0], -1);
    cv::rectangle(img_grid, { kgridh, 0, kgridh, kgridh }, colors[1], -1);
    cv::rectangle(img_grid, { kgridh, kgridh, kgrid - 1, kgrid - 1 }, colors[2], -1);
    cv::rectangle(img_grid, { 0, kgridh, kgridh, kgridh }, colors[3], -1);

    // copy grid into image with border
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
    create_landmark_image(img_grid, dim_grid_fix, 0.0, rcolors, { 0,0,0 }, dpi);

    const int kbx = (kborder * 2) + (img_grid.size().width * xrfix);
    const int kby = (kborder * 2) + (img_grid.size().height * yrfix);

    // set colors of each square in 2x2 grid, index is clockwise from upper left
    cv::Scalar colors[4];
    colors[0] = BGR_COLORS[static_cast<int>(rcolors.c00)];
    colors[1] = BGR_COLORS[static_cast<int>(rcolors.c01)];
    colors[2] = BGR_COLORS[static_cast<int>(rcolors.c11)];
    colors[3] = BGR_COLORS[static_cast<int>(rcolors.c10)];

    // create image that will contain border and grid
    // fill it with border color
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



void BGRLandmark::augment_landmark_image(
    cv::Mat& rimg,
    const int id,
    const int num_dots,
    const double dim_border,
    const double padfac,
    const cv::Scalar border_color,
    const int dpi)
{
    // set limits on number of dots
    if ((num_dots < 1) || (num_dots > 7))
    {
        ///////
        return;
        ///////
    }

    // convert ID to octal digit array which maps to the 8 dot colors
    std::list<int> vec_octal;
    int k = num_dots;
    int id_temp = id;
    while (k--)
    {
        vec_octal.push_front(id_temp % 8);
        id_temp /= 8;
    }

    // set limits on size of border (0 inches to 1 inch)
    double dim_border_fix = dim_border;
    RAIL_MIN(dim_border_fix, 0.0);
    RAIL_MAX(dim_border_fix, 1.0);

    // set limits on radius padding factor
    double padfac_fix = padfac;
    RAIL_MAX(padfac_fix, 1.0);
    RAIL_MIN(padfac_fix, 0.2);

    // input is a landmark image with a border
    const int kborder = static_cast<int>(dim_border_fix * dpi);
    const int kdim = rimg.size().width;
    const int kgrid = (kdim - (2 * kborder));
    const int kstep = kgrid / num_dots;
    
    int krad = ((kgrid / static_cast<int>(num_dots)) / 2);
    krad = static_cast<int>(padfac_fix * krad);

    // make a new image that will hold landmark and colored dots
    // there is additional border padding at the bottom
    cv::Mat img;
    img = cv::Mat::zeros({ kdim, kdim + kstep + kborder }, CV_8UC3);
    cv::rectangle(img, { { 0, 0 }, img.size() }, border_color, -1);

    // draw dots
    int x = (kstep / 2) + kborder;
    for (const auto& r : vec_octal)
    {
        cv::Point ctr = { x, kdim + (kstep / 2) };
        cv::circle(img, ctr, krad, BGR_COLORS[r], -1, cv::LINE_AA);
        x += kstep;
    }

    // draw landmark into new image
    cv::Rect roi = cv::Rect({ 0, 0 }, rimg.size());
    rimg.copyTo(img(roi));

    // assign new image to input image
    rimg = img;
}

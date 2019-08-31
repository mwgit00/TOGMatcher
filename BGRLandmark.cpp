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

#include "opencv2/highgui.hpp"
#include "BGRLandmark.h"



const cv::Scalar bgr_colors[8] =
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



BGRLandmark::BGRLandmark()
{
    init();
    cv::Mat xx;
    cv::Mat xy;
    create_landmark_image(xx, 256, PATTERN_0, 16, { 32,64,128 });
    cv::imwrite("foobgrlm.png", xx);

    create_checkerboard_image(xy, 32, 3, 5);
    cv::imwrite("foobgrcb.png", xy);

}



BGRLandmark::~BGRLandmark()
{

}



void BGRLandmark::init(const int k, const grid_colors_t& rcolors, const int mode)
{
    int fixk;
    int fixkh;

    // set limits on size
    fixk = (k < 3) ? 3 : k;
    fixk = (k > 15) ? 15 : k;
    fixkh = fixk / 2;
    
    // apply mode for template match
    this->mode = mode;

    // stash the template image
    create_template_image(tmpl_bgr, fixk, rcolors);

    // stash offset for this template
    tmpl_offset.width = fixkh;
    tmpl_offset.height = fixkh;
}



void BGRLandmark::perform_match(
    const cv::Mat& rsrc_bgr,
    cv::Mat& rtmatch)
{
    // works well with TM_SQDIFF_NORMED or TM_CCORR_NORMED
    matchTemplate(rsrc_bgr, tmpl_bgr, rtmatch, mode);
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
    create_template_image(t0, k, PATTERN_0);
    create_template_image(t1, k, PATTERN_0N);
    const int xmode = cv::TM_CCOEFF; // TM_CCOEFF, TM_CCORR_NORMED good
    matchTemplate(rsrc, t0, tmatch0, xmode);
    matchTemplate(rsrc, t1, tmatch1, xmode);
    rtmatch = (tmatch0 - tmatch1);
    rtmatch = abs(rtmatch);
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
    colors[0] = bgr_colors[static_cast<int>(rcolors.c00)];
    colors[1] = bgr_colors[static_cast<int>(rcolors.c01)];
    colors[2] = bgr_colors[static_cast<int>(rcolors.c11)];
    colors[3] = bgr_colors[static_cast<int>(rcolors.c10)];

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
    const int k,
    const grid_colors_t& rcolors,
    const int kborder,
    const cv::Scalar border_color)
{
    int xk;
    int xkh;
    int xkb;
    int xkbfix;

    // set limits on 2x2 grid size
    xk = (k < 16) ? 16 : k;
    xk = (k > 256) ? 256 : k;
    xkh = xk / 2;
    
    // set limits on border size
    xkbfix = (kborder < 0) ? 0 : kborder;
    xkbfix = (kborder > 16) ? 16 : kborder;
    xkb = xk + (xkbfix * 2);

    // set colors of each square in 2x2 grid, index is clockwise from upper left
    cv::Scalar colors[4];
    colors[0] = bgr_colors[static_cast<int>(rcolors.c00)];
    colors[1] = bgr_colors[static_cast<int>(rcolors.c01)];
    colors[2] = bgr_colors[static_cast<int>(rcolors.c11)];
    colors[3] = bgr_colors[static_cast<int>(rcolors.c10)];

    // create image that will contain border and grid
    // fill it with border color
    rimg = cv::Mat::zeros({ xkb, xkb }, CV_8UC3);
    cv::rectangle(rimg, { 0, 0, xkb, xkb }, border_color, -1);

    // create image with just the grid
    cv::Mat img_grid = cv::Mat::zeros({ xk, xk }, CV_8UC3);

    cv::Rect roi = cv::Rect({ xkbfix, xkbfix }, img_grid.size());

    // fill in 2x2 blocks (clockwise from upper left)
    cv::rectangle(img_grid, { 0, 0, xkh - 1, xkh - 1 }, colors[0], -1);
    cv::rectangle(img_grid, { xkh, 0, xkh, xkh }, colors[1], -1);
    cv::rectangle(img_grid, { xkh, xkh, xk - 1, xk - 1 }, colors[2], -1);
    cv::rectangle(img_grid, { 0, xkh, xkh, xkh }, colors[3], -1);

    // copy grid into image with border
    img_grid.copyTo(rimg(roi));
}



void BGRLandmark::create_checkerboard_image(
    cv::Mat& rimg,
    const int k,
    const int xrepeat,
    const int yrepeat,
    const grid_colors_t& rcolors,
    const int kborder,
    const cv::Scalar border_color)
{
    int xkbx;
    int xkby;
    int xkbfix;
    int xrfix;
    int yrfix;

    cv::Mat img_grid;
    create_landmark_image(img_grid, k, rcolors, 0);

    // set limits on border size
    xkbfix = (kborder < 0) ? 0 : kborder;
    xkbfix = (kborder > 16) ? 16 : kborder;

    xrfix = (xrepeat < 2) ? 2 : xrepeat;
    xrfix = (xrepeat > 8) ? 8 : xrepeat;
    yrfix = (yrepeat < 2) ? 2 : yrepeat;
    yrfix = (yrepeat > 8) ? 8 : yrepeat;

    xkbx = (xkbfix * 2) + (img_grid.size().width * xrfix);
    xkby = (xkbfix * 2) + (img_grid.size().height * yrfix);

    // set colors of each square in 2x2 grid, index is clockwise from upper left
    cv::Scalar colors[4];
    colors[0] = bgr_colors[static_cast<int>(rcolors.c00)];
    colors[1] = bgr_colors[static_cast<int>(rcolors.c01)];
    colors[2] = bgr_colors[static_cast<int>(rcolors.c11)];
    colors[3] = bgr_colors[static_cast<int>(rcolors.c10)];

    // create image that will contain border and grid
    // fill it with border color
    rimg = cv::Mat::zeros({ xkbx, xkby }, CV_8UC3);
    cv::rectangle(rimg, { 0, 0, xkbx, xkby }, border_color, -1);

    // repeat the block pattern
    for (int j = 0; j < yrfix; j++)
    {
        for (int i = 0; i < xrfix; i++)
        {
            cv::Rect roi = { (i * k) + xkbfix, (j * k) + xkbfix, k, k };
            img_grid.copyTo(rimg(roi));
        }
    }
}

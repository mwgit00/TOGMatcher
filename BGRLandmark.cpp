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
    create_landmark_image(xx, 256, { bgr_t::BLACK, bgr_t::YELLOW, bgr_t::BLACK, bgr_t::CYAN }, 16, { 32,64,128 });
}



BGRLandmark::~BGRLandmark()
{

}



void BGRLandmark::init(const int k, const grid_colors_t& r, const int mode)
{
	int xk;
	int xkh;

    this->mode = mode;

    // set limits on size
	xk = (k < 3) ? 3 : k;
	xk = (k > 15) ? 15 : k;
	xkh = xk / 2;

    // set colors of each square in 2x2 grid, index is clockwise from upper left
    cv::Scalar colors[4];
    colors[0] = bgr_colors[static_cast<int>(r.c00)];
    colors[1] = bgr_colors[static_cast<int>(r.c01)];
    colors[2] = bgr_colors[static_cast<int>(r.c11)];
    colors[3] = bgr_colors[static_cast<int>(r.c10)];

    tmpl_bgr = cv::Mat::zeros({ xk, xk }, CV_8UC3);

    // fill in 2x2 blocks (clockwise from upper left)
    cv::rectangle(tmpl_bgr, { 0, 0, xkh, xkh }, colors[0], -1);
    cv::rectangle(tmpl_bgr, { xkh + 1, 0, xkh, xkh }, colors[1], -1);
    cv::rectangle(tmpl_bgr, { xkh, xkh, xk - 1, xk - 1 }, colors[2], -1);
    cv::rectangle(tmpl_bgr, { 0, xkh + 1, xkh, xkh }, colors[3], -1);
    
    // fill in average at borders between blocks
    cv::Scalar avg_c00_c10 = (colors[0] + colors[1]) / 2;
    cv::Scalar avg_c10_c11 = (colors[1] + colors[2]) / 2;
    cv::Scalar avg_c11_c01 = (colors[2] + colors[3]) / 2;
    cv::Scalar avg_c01_c00 = (colors[3] + colors[0]) / 2;
    cv::line(tmpl_bgr, { xkh, 0 }, { xkh, xkh }, avg_c00_c10);
    cv::line(tmpl_bgr, { xkh, xkh }, { xk - 1, xkh }, avg_c10_c11);
    cv::line(tmpl_bgr, { xkh, xkh }, { xkh, xk - 1 }, avg_c11_c01);
    cv::line(tmpl_bgr, { 0, xkh }, { xkh, xkh }, avg_c01_c00);

    // fill in average of all blocks at central point
    cv::Scalar avg_all = (colors[0] + colors[1] + colors[2] + colors[3]) / 4;
    cv::line(tmpl_bgr, { xkh, xkh }, { xkh, xkh }, avg_all);

    tmpl_offset.width = xk / 2;
    tmpl_offset.height = xk / 2;

    cv::imwrite("foobgr.png", tmpl_bgr);
}



void BGRLandmark::perform_match(
    const cv::Mat& rsrc_bgr,
    cv::Mat& rtmatch)
{
    // works well with TM_SQDIFF_NORMED or TM_CCORR_NORMED
    matchTemplate(rsrc_bgr, tmpl_bgr, rtmatch, mode);
}



void BGRLandmark::create_landmark_image(
    cv::Mat& rimg,
    const int k,
    const grid_colors_t& r,
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
    colors[0] = bgr_colors[static_cast<int>(r.c00)];
    colors[1] = bgr_colors[static_cast<int>(r.c01)];
    colors[2] = bgr_colors[static_cast<int>(r.c11)];
    colors[3] = bgr_colors[static_cast<int>(r.c10)];

    // create image that will contain border and grid
    // fill it with border color
    rimg = cv::Mat::zeros({ xkb, xkb }, CV_8UC3);
    cv::rectangle(rimg, { 0, 0, xkb, xkb }, border_color, -1);

    // create image with just the grid
    cv::Mat img_grid = cv::Mat::zeros({ xk, xk }, CV_8UC3);

    cv::Rect roi = cv::Rect({ xkbfix, xkbfix }, img_grid.size());

    // fill in 2x2 blocks (clockwise from upper left)
    cv::rectangle(img_grid, { 0, 0, xkh, xkh }, colors[0], -1);
    cv::rectangle(img_grid, { xkh + 1, 0, xkh, xkh }, colors[1], -1);
    cv::rectangle(img_grid, { xkh, xkh, xk - 1, xk - 1 }, colors[2], -1);
    cv::rectangle(img_grid, { 0, xkh + 1, xkh, xkh }, colors[3], -1);

    // copy grid into image with border
    img_grid.copyTo(rimg(roi));

    cv::imwrite("foobgrx.png", rimg);
}

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
#include "TOGMatcher.h"


const int TEMPLATE_DEPTH = CV_32F;


TOGMatcher::TOGMatcher() :
    tmpl_offset({ 0,0 })
{
}


TOGMatcher::~TOGMatcher()
{
}


void TOGMatcher::create_template_from_file(const char * s, const int ksize, const double mag_thr)
{
    cv::Mat tmplsrc = cv::imread(s, cv::IMREAD_GRAYSCALE);
    create_templates(tmplsrc, ksize, mag_thr);
}


void TOGMatcher::create_template_from_img(const cv::Mat& rsrc, const int ksize, const double mag_thr)
{
    create_templates(rsrc, ksize, mag_thr);
}


void TOGMatcher::create_templates(const cv::Mat& rsrc, const int ksize, const double mag_thr)
{
    double qmax;
    cv::Mat temp_m;
    cv::Mat temp_a;
    cv::Mat temp_mask;
    std::vector<cv::Point> all_pts;

    // calculate X and Y gradients
    // they will become the gradient template images
    Sobel(rsrc, tmpl_dx, TEMPLATE_DEPTH, 1, 0, ksize);
    Sobel(rsrc, tmpl_dy, TEMPLATE_DEPTH, 0, 1, ksize);

    // create gradient magnitude mask
    // everything above the threshold (a fraction of the max) will be considered valid
    cv::cartToPolar(tmpl_dx, tmpl_dy, temp_m, temp_a);
    cv::minMaxLoc(temp_m, nullptr, &qmax);
    temp_mask = (temp_m > (qmax * mag_thr));

    // find external contour of magnitude mask
    // then find minimal bounding box around it
    cv::findContours(temp_mask, src_contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
    for (size_t i = 0; i < src_contours.size(); i++)
    {
        all_pts.insert(all_pts.end(), src_contours[i].begin(), src_contours[i].end());
    }
    cv::Rect rbox = cv::boundingRect(all_pts);

    // shrink the magnitude mask so it fits in the minimal bounding box
    // there is no need to include any pixels outside this box
    temp_mask = temp_mask(rbox);

    // convert CV_8U mask to CV_32F mask (scale 0-255 to 0.0-1.0)
    temp_mask.convertTo(tmpl_mask_32F, TEMPLATE_DEPTH, 1.0 / 255.0);

    // shrink the dX and dY templates so they fit in the minimal bounding box
    // and apply mask to zero the pixels for gradients with small magnitudes
    tmpl_dx = tmpl_dx(rbox);
    tmpl_dy = tmpl_dy(rbox);
    tmpl_dx.mul(tmpl_mask_32F);
    tmpl_dy.mul(tmpl_mask_32F);

    // must update the external contour after resizing
    cv::findContours(temp_mask, src_contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    tmpl_offset = temp_mask.size();
    tmpl_offset.x /= 2;
    tmpl_offset.y /= 2;
}


void TOGMatcher::perform_match(
    const cv::Mat& rsrc,
    cv::Mat& rtmatch,
    const bool is_mask_enabled,
    const int ksize)
{
    cv::Mat grad_x;
    cv::Mat grad_y;
    cv::Mat tmatch_x;
    cv::Mat tmatch_y;

    // calculate X and Y gradient images
    Sobel(rsrc, grad_x, TEMPLATE_DEPTH, 1, 0, ksize);
    Sobel(rsrc, grad_y, TEMPLATE_DEPTH, 0, 1, ksize);

    // perform match with dX and dY magnitude templates
    // it is up to the user whether or not the mask is enabled
    if (is_mask_enabled)
    {
        matchTemplate(grad_x, tmpl_dx, tmatch_x, cv::TM_CCORR_NORMED, tmpl_mask_32F);
        matchTemplate(grad_y, tmpl_dy, tmatch_y, cv::TM_CCORR_NORMED, tmpl_mask_32F);
    }
    else
    {
        matchTemplate(grad_x, tmpl_dx, tmatch_x, cv::TM_CCORR_NORMED);
        matchTemplate(grad_y, tmpl_dy, tmatch_y, cv::TM_CCORR_NORMED);
    }

    // combine results by multiplying both matches together
    rtmatch = tmatch_x.mul(tmatch_y);
}


void TOGMatcher::perform_match_sqdiff(
    const cv::Mat& rsrc,
    cv::Mat& rtmatch,
    const bool is_mask_enabled,
    const int ksize)
{
    cv::Mat grad_x;
    cv::Mat grad_y;
    cv::Mat tmatch_x;
    cv::Mat tmatch_y;

    // calculate X and Y gradient images
    Sobel(rsrc, grad_x, TEMPLATE_DEPTH, 1, 0, ksize);
    Sobel(rsrc, grad_y, TEMPLATE_DEPTH, 0, 1, ksize);

    // perform match with dX and dY magnitude templates
    // it is up to the user whether or not the mask is enabled
    if (is_mask_enabled)
    {
        matchTemplate(grad_x, tmpl_dx, tmatch_x, cv::TM_SQDIFF, tmpl_mask_32F);
        matchTemplate(grad_y, tmpl_dy, tmatch_y, cv::TM_SQDIFF, tmpl_mask_32F);
    }
    else
    {
        matchTemplate(grad_x, tmpl_dx, tmatch_x, cv::TM_SQDIFF);
        matchTemplate(grad_y, tmpl_dy, tmatch_y, cv::TM_SQDIFF);
    }

    // combine results by adding match results
    // best results for SQDIFF are minimums so do a sign flip
    rtmatch = -(tmatch_x + tmatch_y);
}

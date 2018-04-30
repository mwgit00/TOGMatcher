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

#ifndef TOG_MATCHER_H_
#define TOG_MATCHER_H_

#include "opencv2/imgproc.hpp"


// Selects size for use in Sobel filter
// Options are -1 for Scharr or 1, 3, 5, 7
#define TOG_DEFAULT_KSIZE   (-1)

// Threshold factor for gradient magnitude mask used with templates
// 0.1 to 0.25 is typical range
#define TOG_DEFAULT_MAG_THR (0.1)


class TOGMatcher
{
public:
    
    TOGMatcher();
    virtual ~TOGMatcher();

    void create_template_from_file(
        const char * s,
        const int ksize = TOG_DEFAULT_KSIZE,
        const double mag_thr = TOG_DEFAULT_MAG_THR);
    
    void create_template_from_img(
        const cv::Mat& rsrc,
        const int ksize = TOG_DEFAULT_KSIZE,
        const double mag_thr = TOG_DEFAULT_MAG_THR);
    
    void perform_match(
        const cv::Mat& rsrc,
        cv::Mat& rtmatch,
        const bool is_mask_enabled = true,
        const int ksize = TOG_DEFAULT_KSIZE);

    const cv::Mat& get_template_mask(void) const { return tmpl_mask_32F; }
    const cv::Mat& get_template_dx(void) const { return tmpl_dx; }
    const cv::Mat& get_template_dy(void) const { return tmpl_dy; }
    const std::vector<std::vector<cv::Point>>& get_contours() { return src_contours; }

private:

    void create_templates(
        const cv::Mat& rsrc,
        const int ksize,
        const double mag_thr);

    // Gradient magnitude mask for template
    cv::Mat tmpl_mask_32F;  
    
    // Sobel 1st order dX template
    cv::Mat tmpl_dx;
    
    // Sobel 1st order dY template
    cv::Mat tmpl_dy;

    // Contour(s) of template that can be drawn onto an image
    std::vector<std::vector<cv::Point>> src_contours;
};

#endif // TOG_MATCHER_H_

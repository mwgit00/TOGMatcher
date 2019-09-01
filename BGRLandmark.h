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

#ifndef BGR_LANDMARK_
#define BGR_LANDMARK_

#include "opencv2/imgproc.hpp"


class BGRLandmark
{
public:

    // names of colors with max and/or min BGR components
    enum class bgr_t : int
    {
        BLACK,
        RED,
        GREEN,
        YELLOW,
        BLUE,
        MAGENTA,
        CYAN,
        WHITE
    };

    // there 8 colors with max/min BGR components
    static const cv::Scalar bgr_colors[8];


    // codes for the color of each block in 2x2 grid (clockwise from upper left)
    typedef struct zub
    {
        bgr_t c00;
        bgr_t c01;
        bgr_t c11;
        bgr_t c10;
        zub() {}
        zub(bgr_t a, bgr_t b, bgr_t c, bgr_t d) : c00(a), c01(b), c11(c), c10(d) {}
    } grid_colors_t;

    // a default checkerboard will have one of these patterns at each 2x2 corner
    static const grid_colors_t PATTERN_0;
    static const grid_colors_t PATTERN_1;
    static const grid_colors_t PATTERN_2;
    static const grid_colors_t PATTERN_3;


    BGRLandmark();
	virtual ~BGRLandmark();

    
    void init(
        const int k = 5,
        const grid_colors_t& rcolors = PATTERN_0,
        const int mode = cv::TM_CCOEFF);

	void perform_match(
		const cv::Mat& rsrc,
		cv::Mat& rtmatch);

    void perform_match_cb(
        const cv::Mat& rsrc,
        cv::Mat& rtmatch);

    const cv::Size& get_template_offset(void) const { return tmpl_offset; }

    
    static bgr_t invert_bgr(const bgr_t color);
    static grid_colors_t invert_grid_colors(const grid_colors_t& rcolors);
    
    
    // creates a 2x2 BGR template of pixel dimension k
    static void create_template_image(
        cv::Mat& rimg,
        const int k,
        const grid_colors_t& rcolors);

    
    static void create_landmark_image(
        cv::Mat& rimg,
        const double dim_grid = 1.0,
        const double dim_border = 0.25,
        const grid_colors_t& rcolors = PATTERN_0,
        const cv::Scalar border_color = { 128, 128, 128 },
        const int dpi = 96);

    static void create_checkerboard_image(
        cv::Mat& rimg,
        const int xrepeat,
        const int yrepeat,
        const double dim_grid = 2.0,
        const double dim_border = 0.25,
        const grid_colors_t& rcolors = PATTERN_0,
        const cv::Scalar border_color = { 128, 128, 128 },
        const int dpi = 96);

    static void augment_landmark_image(
        cv::Mat& rimg,
        const double dim_border,
        const double padfac,
        const std::vector<bgr_t>& rvec,
        const cv::Scalar border_color = { 128, 128, 128 },
        const int dpi = 96);

private:

    // mode for matchTemplate
    int mode;
    
    // the 2x2 grid BGR template
    cv::Mat tmpl_bgr;

    // offset for centering template location
    cv::Size tmpl_offset;
};

#endif // BGR_LANDMARK_
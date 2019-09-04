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
    static const cv::Scalar BGR_COLORS[8];
    static const cv::Scalar BGR_TO_HLS[8];
    static const cv::Scalar BGR_TO_HSV[8];
    static const cv::Scalar BGR_BORDER;


    // codes for the color of each block in 2x2 grid (clockwise from upper left)
    typedef struct zub
    {
        bgr_t c00;
        bgr_t c01;
        bgr_t c11;
        bgr_t c10;
    } grid_colors_t;

    // a default checkerboard will have one of these patterns at each 2x2 corner
    static const grid_colors_t PATTERN_0;
    static const grid_colors_t PATTERN_1;
    static const grid_colors_t PATTERN_2;
    static const grid_colors_t PATTERN_3;


    BGRLandmark();
	virtual ~BGRLandmark();

    
    void init(
        const int k = 11,
        const grid_colors_t& rcolors = PATTERN_0,
        const int mode = cv::TM_CCOEFF);

    void perform_match(
        const cv::Mat& rsrc,
        cv::Mat& rtmatch,
        std::vector<std::vector<cv::Point>>& rcontours,
        std::vector<cv::Point>& rpts,
        double * pmax = nullptr,
        cv::Point * ppt = nullptr);

    void perform_match_gray(
        const cv::Mat& rsrc,
        cv::Mat& rtmatch,
        std::vector<std::vector<cv::Point>>& rcontours,
        std::vector<cv::Point>& rpts,
        double* pmax = nullptr,
        cv::Point* ppt = nullptr);


    void perform_match_cb(
        const cv::Mat& rsrc,
        cv::Mat& rtmatch);

    const cv::Size& get_template_offset(void) const { return tmpl_offset; }

    double check_grid_hues(const cv::Mat& rimg, const cv::Point& rpt) const;

    
    // returns color index that is inverse of input color index
    static bgr_t invert_bgr(const bgr_t color);
    
    // inverts the color indexes for a grid
    static grid_colors_t invert_grid_colors(const grid_colors_t& rcolors);
    
    
    // creates printable 2x2 landmark image
    static void create_landmark_image(
        cv::Mat& rimg,
        const double dim_grid = 1.0,
        const double dim_border = 0.25,
        const grid_colors_t& rcolors = PATTERN_0,
        const cv::Scalar border_color = BGR_BORDER,
        const int dpi = 96);

    // creates printable checkerboard by repeating a 2x2 landmark pattern
    static void create_checkerboard_image(
        cv::Mat& rimg,
        const int xrepeat,
        const int yrepeat,
        const double dim_grid = 2.0,
        const double dim_border = 0.25,
        const grid_colors_t& rcolors = PATTERN_0,
        const cv::Scalar border_color = BGR_BORDER,
        const int dpi = 96);

    // converts ID to BGR dots and puts dots at bottom of printable 2x2 landmark image
    static void augment_landmark_image(
        cv::Mat& rimg,
        const int id,
        const int num_dots = 3,
        const double dim_border = 0.25,
        const double padfac = 0.75,
        const cv::Scalar border_color = BGR_BORDER,
        const int dpi = 96);


private:
    
    // creates a 2x2 BGR template of pixel dimension k
    static void create_template_image(
        cv::Mat& rimg,
        const int k,
        const grid_colors_t& rcolors);


private:

    // 2x2 grid or circular "X" pattern
    bool is_grid_pattern;

    // mode for matchTemplate
    int mode;

    // pattern for the 2x2 grid
    grid_colors_t pattern;

    // threshold for match consideration
    double match_thr;

    // score when matching template against itself
    double ideal_score;
    
    // the 2x2 grid BGR template
    cv::Mat tmpl_bgr;

    // the 2x2 grid hue template and mask
    cv::Mat tmpl_hue;
    cv::Mat tmpl_hue_mask;

    cv::Mat tmpl_gray_p;
    cv::Mat tmpl_gray_n;

    // offset for centering template location
    cv::Size tmpl_offset;
};

#endif // BGR_LANDMARK_
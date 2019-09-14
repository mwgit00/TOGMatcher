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
#include "TOGMatcher.h"


class BGRLandmark
{
public:

    typedef struct
    {
        cv::Point ctr;      // center point of landmark
        double diff;        // value of positive template match minus negative template match
        double rng;         // range of pixels in candidate ROI
        double min;         // min pixel in candidate ROI
    } landmark_info_t;

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
    static const cv::Scalar BGR_BORDER;


    // codes for the color of each block in 2x2 grid (clockwise from upper left)
    typedef struct zub
    {
        bgr_t c00;
        bgr_t c01;
        bgr_t c11;
        bgr_t c10;
    } grid_colors_t;

    // some standard colored checkerboard patterns
    static const grid_colors_t PATTERN_0;
    static const grid_colors_t PATTERN_A;
    static const grid_colors_t PATTERN_B;
    static const grid_colors_t PATTERN_C;
    static const grid_colors_t PATTERN_D;
    static const grid_colors_t PATTERN_E;
    static const grid_colors_t PATTERN_F;


    BGRLandmark();
	virtual ~BGRLandmark();

    
    void init(
        const int k = 9,
        const double thr_corr = 1.6,    // threshold for dual match which produces results in range 0 to 2
        const int thr_pix_rng = 85,     // seek range of pixels in landmark > 1/3 of max pixel val (255)
        const int thr_pix_min = 85);    // seek dark regions in landmark < 1/3 of max pixel val (255)

    void perform_match(
        const cv::Mat& rsrc,
        cv::Mat& rtmatch,
        std::vector<BGRLandmark::landmark_info_t>& rpts);

    const cv::Point& get_template_offset(void) const { return tmpl_offset; }

    int identify_colors(const cv::Mat& rimg, const BGRLandmark::landmark_info_t& rinfo) const;

    
    // creates printable 2x2 landmark image
    static void create_landmark_image(
        cv::Mat& rimg,
        const double dim_grid = 1.0,
        const double dim_border = 0.25,
        const grid_colors_t& rcolors = PATTERN_A,
        const cv::Scalar border_color = BGR_BORDER,
        const int dpi = 96);

    // creates printable checkerboard by repeating a 2x2 landmark pattern
    static void create_checkerboard_image(
        cv::Mat& rimg,
        const int xrepeat,
        const int yrepeat,
        const double dim_grid = 2.0,
        const double dim_border = 0.25,
        const grid_colors_t& rcolors = PATTERN_A,
        const cv::Scalar border_color = BGR_BORDER,
        const int dpi = 96);


private:
    
    // creates a 2x2 grid or "X" pattern BGR template of pixel dimension k
    static void create_template_image(
        cv::Mat& rimg,
        const int k,
        const grid_colors_t& rcolors);


private:

    // thresholds for match consideration
    double thr_corr;
    int thr_pix_rng;
    int thr_pix_min;

    // templates for 2x2 checkerboard grid
    cv::Mat tmpl_gray_p;
    cv::Mat tmpl_gray_n;

    // offset for centering template location
    cv::Point tmpl_offset;
};

#endif // BGR_LANDMARK_

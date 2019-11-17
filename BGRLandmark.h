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

#ifndef BGR_LANDMARK_H_
#define BGR_LANDMARK_H_

// uncomment line below to collect up to 1000 samples and write them to an image file when done
#define _COLLECT_SAMPLES

#include <map>
#include "opencv2/imgproc.hpp"
#include "DCTFeature.h"
//#include "TOGMatcher.h"


class BGRLandmark
{
public:

    typedef struct
    {
        cv::Point ctr;      // center point of landmark
        double diff;        // value of positive template match minus negative template match
        double rng;         // range of pixels in candidate ROI
        double min;         // min pixel in candidate ROI
        int code;           // color code, -1 for unknown, else 0-11
    } landmark_info_t;

    // names of colors with 0 or 255 as the BGR components
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

    // codes for the color of each block in 2x2 grid (clockwise from upper left)
    typedef struct _T_grid_colors_struct
    {
        bgr_t c00;
        bgr_t c01;
        bgr_t c11;
        bgr_t c10;
    } grid_colors_t;

    // there are 8 colors that only have 0 or 255 as the BGR components
    static const cv::Scalar BGR_COLORS[8];
    static const cv::Scalar BGR_BORDER;

    // the supported landmark color patterns
    static const std::map<char, grid_colors_t> PATTERN_MAP;

public:

    BGRLandmark();
    virtual ~BGRLandmark();
    
    // init with "good" default settings
    void init(
        const int k = 9,
        const double thr_corr = 1.6,    // threshold for dual match (range is 0.0 to 2.0)
        const int thr_pix_rng = 40,     // grey image pixel range threshold for pre-proc
        const int thr_pix_min = 70,     // grey image dark pixel threshold for pre-proc
        const int thr_bgr_rng = 20);    // range in BGR required for color matching step

    // runs the match on an original BGR image and possibly pre-processed gray image
    // it returns a gray image with the raw template match and a vector of landmark info
    void perform_match(
        const cv::Mat& rsrc_bgr,
        const cv::Mat& rsrc,
        cv::Mat& rtmatch,
        std::vector<BGRLandmark::landmark_info_t>& rpts);

    // gets centering offset for the landmark template
    const cv::Point& get_template_offset(void) const { return tmpl_offset; }

    // normally color ID should always be enabled
    // but it can be turned off for testing
    void set_color_id_enable(const bool f) { is_color_id_enabled = f; }

    
    // creates printable 2x2 landmark image
    static void create_landmark_image(
        cv::Mat& rimg,
        const double dim_grid = 1.0,
        const double dim_border = 0.25,
        const grid_colors_t& rcolors = { bgr_t::BLACK, bgr_t::WHITE, bgr_t::BLACK, bgr_t::WHITE },
        const cv::Scalar border_color = BGR_BORDER,
        const int dpi = 96);

    // creates printable multi-landmark image by repeating 2x2 landmark patterns
    // they are placed in row-major order in the image based on the repeat counts
    // the labels identify each landmark, cycling back around if necessary
    // it's up to the user to pick sane dimensions and repeat counts
    static void create_multi_landmark_image(
        cv::Mat& rimg,
        const std::string& rslabels,
        const int xrepeat,
        const int yrepeat,
        const double dim_grid = 1.0,
        const double dim_spacing = 2.0,
        const double dim_border = 0.25,
        const cv::Scalar border_color = BGR_BORDER,
        const int dpi = 96);

    static int compare_by_code(BGRLandmark::landmark_info_t& a, BGRLandmark::landmark_info_t& b)
        { return a.code < b.code; }

private:

    // creates a 2x2 grid BGR template of pixel dimension k
    static void create_template_image(
        cv::Mat& rimg,
        const int k,
        const grid_colors_t& rcolors);

    // takes landmark info and snapshot of landmark
    // and tries to identify the colors in the non-black squares
    void identify_colors(const cv::Mat& rimg, BGRLandmark::landmark_info_t& rinfo) const;

    // converts the "sign" of the landmark and its 2 bright colors into a single code
    static int get_bgr_code(double s, const int a, const int b);


private:

    // size of square landmark template
    int kdim;

    // thresholds for match consideration
    double thr_corr;
    int thr_pix_rng;
    int thr_pix_min;
    int thr_bgr_rng;

    // templates for 2x2 checkerboard grid
    cv::Mat tmpl_gray_p;
    cv::Mat tmpl_gray_n;

    // offset for centering template location
    cv::Point tmpl_offset;

    // flag for controlling color ID function
    bool is_color_id_enabled;

#ifdef _COLLECT_SAMPLES
public:
    const int sampx = 40;
    const int sampy = 25;
    int samp_ct;
    cv::Mat samples;
#endif

    DCTFeature dct_fv;
};

#endif // BGR_LANDMARK_H_

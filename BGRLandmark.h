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

    // names of colors with max or min BGR components
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
    typedef struct
    {
        bgr_t c00;
        bgr_t c01;
        bgr_t c11;
        bgr_t c10;
    } grid_colors_t;

    const grid_colors_t PATTERN_0 = { bgr_t::BLACK, bgr_t::YELLOW, bgr_t::BLACK, bgr_t::CYAN };
    const grid_colors_t PATTERN_1 = { bgr_t::CYAN, bgr_t::BLACK, bgr_t::YELLOW, bgr_t::BLACK };
	
    BGRLandmark();
	virtual ~BGRLandmark();

    void init(
        const int k = 7,
        const grid_colors_t& r = { bgr_t::BLACK, bgr_t::YELLOW, bgr_t::BLACK, bgr_t::CYAN },
        const int mode = cv::TM_SQDIFF_NORMED);

	void perform_match(
		const cv::Mat& rsrc,
		cv::Mat& rtmatch);

    const cv::Size& get_template_offset(void) const { return tmpl_offset; }

    static void create_landmark_image(
        cv::Mat& rimg,
        const int k,
        const grid_colors_t& r,
        const int kborder = 4,
        const cv::Scalar border_color = { 128, 128, 128 });

private:

    // mode for matchTemplate
    int mode;
    
    // the 2x2 grid BGR template
    cv::Mat tmpl_bgr;

    // offset for centering template location
    cv::Size tmpl_offset;
};

#endif // BGR_LANDMARK_
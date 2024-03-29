// MIT License
//
// Copyright(c) 2021 Mark Whitney
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
#include <iostream>
#include "opencv2/highgui.hpp"
#include "BGRLandmark.h"


namespace cpoz
{
    const cv::Scalar BGRLandmark::BGR_COLORS[8] =
    {
        cv::Scalar(0, 0, 0),        // black
        cv::Scalar(0, 0, 255),      // red
        cv::Scalar(0, 255, 0),      // green
        cv::Scalar(0, 255, 255),    // yellow   (0,G,R)
        cv::Scalar(255, 0, 0),      // blue
        cv::Scalar(255, 0, 255),    // magenta  (B,0,R)
        cv::Scalar(255, 255, 0),    // cyan     (B,G,0)
        cv::Scalar(255, 255, 255),  // white
    };

    const cv::Scalar BGRLandmark::BGR_BORDER = { 128, 128, 128 };

    const std::map<char, BGRLandmark::grid_colors_t> BGRLandmark::PATTERN_MAP =
    {
        { '0', { bgr_t::BLACK, bgr_t::WHITE, bgr_t::BLACK, bgr_t::WHITE }},
        { '1', { bgr_t::WHITE, bgr_t::BLACK, bgr_t::WHITE, bgr_t::BLACK }},
        { 'A', { bgr_t::BLACK, bgr_t::YELLOW, bgr_t::BLACK, bgr_t::MAGENTA }},
        { 'B', { bgr_t::BLACK, bgr_t::YELLOW, bgr_t::BLACK, bgr_t::CYAN }},
        { 'C', { bgr_t::BLACK, bgr_t::MAGENTA, bgr_t::BLACK, bgr_t::YELLOW }},
        { 'D', { bgr_t::BLACK, bgr_t::MAGENTA, bgr_t::BLACK, bgr_t::CYAN }},
        { 'E', { bgr_t::BLACK, bgr_t::CYAN, bgr_t::BLACK, bgr_t::YELLOW }},
        { 'F', { bgr_t::BLACK, bgr_t::CYAN, bgr_t::BLACK, bgr_t::MAGENTA }},
        { 'G', { bgr_t::YELLOW, bgr_t::BLACK, bgr_t::MAGENTA, bgr_t::BLACK }},
        { 'H', { bgr_t::YELLOW, bgr_t::BLACK, bgr_t::CYAN, bgr_t::BLACK }},
        { 'I', { bgr_t::MAGENTA, bgr_t::BLACK, bgr_t::YELLOW, bgr_t::BLACK }},
        { 'J', { bgr_t::MAGENTA, bgr_t::BLACK, bgr_t::CYAN, bgr_t::BLACK }},
        { 'K', { bgr_t::CYAN, bgr_t::BLACK, bgr_t::YELLOW, bgr_t::BLACK }},
        { 'L', { bgr_t::CYAN, bgr_t::BLACK, bgr_t::MAGENTA, bgr_t::BLACK }}
    };

    const std::string BGRLandmark::CALIB_LABELS = "ADGJBEHKCFIL";



    // returns a value "railed" to fall within a max-min range
    template <class T>
    static T apply_rail(const T v, const T vmin, const T vmax)
    {
        return (v > vmax) ? vmax : ((v < vmin) ? vmin : v);
    }



    static double bgr_to_gray(const cv::Vec3b& rv)
    {
        // gray = 0.299 R + 0.587 G + 0.114 B
        return (rv[0] * 0.114 + rv[1] * 0.587 + rv[2] * 0.299);
    }



    BGRLandmark::BGRLandmark()
    {
        init();
    }



    BGRLandmark::~BGRLandmark()
    {
#ifdef _COLLECT_SAMPLES
        cv::imwrite("samples_1K.png", samples);
#endif
    }



    void BGRLandmark::init(
        const int k,
        const double thr_corr,
        const int thr_pix_rng,
        const int thr_pix_min,
        const int thr_bgr_rng,
        const double thr_sqdiff)
    {
        // fix k to be odd and in range 7-15
        int fixk = ((k / 2) * 2) + 1;
        kdim = apply_rail<int>(fixk, 7, 15);

        // apply thresholds
        // TODO -- assert type is CV_8U somewhere during match
        this->thr_corr = thr_corr;
        this->thr_pix_rng = thr_pix_rng;
        this->thr_pix_min = thr_pix_min;
        this->thr_bgr_rng = thr_bgr_rng;
        this->thr_sqdiff = thr_sqdiff;

        // create the B&W matching templates
        cv::Mat tmpl_bgr;
        create_template_image(tmpl_bgr, kdim, PATTERN_MAP.find('0')->second);
        cv::cvtColor(tmpl_bgr, tmpl_gray_p, cv::COLOR_BGR2GRAY);
        create_template_image(tmpl_bgr, kdim, PATTERN_MAP.find('1')->second);
        cv::cvtColor(tmpl_bgr, tmpl_gray_n, cv::COLOR_BGR2GRAY);

        // stash offset for this template
        const int fixkh = kdim / 2;
        tmpl_offset.x = fixkh;
        tmpl_offset.y = fixkh;

        is_color_id_enabled = true;

#ifdef _COLLECT_SAMPLES
        samp_ct = 0;
        samples = cv::Mat::zeros({ (kdim + 4) * sampx, (kdim + 4) * sampy }, CV_8UC3);
#endif
    }



    void BGRLandmark::perform_match(
        const cv::Mat& rsrc_bgr,
        const cv::Mat& rsrc,
        cv::Mat& rtmatch,
        std::vector<BGRLandmark::landmark_info_t>& rinfo)
    {
        const int xmode = cv::TM_CCOEFF_NORMED;

        // match the template
        // good match will be close to +1.0 or -1.0
        // so take absolute value of result
        cv::Mat tmatch;
        matchTemplate(rsrc, tmpl_gray_p, tmatch, xmode);
        rtmatch = abs(tmatch);

        // find local maxima in the match results...
        cv::Mat maxima_mask;
        cv::dilate(rtmatch, maxima_mask, cv::Mat());
        cv::compare(rtmatch, maxima_mask, maxima_mask, cv::CMP_GE);

        // then apply absolute threshold to get the best local maxima
        cv::Mat match_masked = (rtmatch > thr_corr);
        maxima_mask = maxima_mask & match_masked;

        // collect point locations of all local maxima
        std::vector<cv::Point> vec_maxima_pts;
        cv::findNonZero(maxima_mask, vec_maxima_pts);

        // check each maxima...
        for (const auto& rpt : vec_maxima_pts)
        {
            // positive means black in upper-left/lower-right
            // negative means black in lower-left/upper-right
            float corr = tmatch.at<float>(rpt);

            // extract gray region of interest
            const cv::Rect roi = cv::Rect(rpt, tmpl_gray_p.size());
            cv::Mat img_roi(rsrc(roi));

            // get gray pixel range stats in ROI
            double min_roi;
            double max_roi;
            cv::minMaxLoc(img_roi, &min_roi, &max_roi);
            double rng_roi = max_roi - min_roi;

            // a landmark ROI should have two dark squares and and two light squares
            // see if ROI has large range in pixel values and a minimum that is sufficiently dark
            if ((rng_roi >= thr_pix_rng) && (min_roi <= thr_pix_min))
            {
                // start filling in landmark info
                landmark_info_t lminfo{ rpt + tmpl_offset, corr, rng_roi, min_roi, -1, 0.0 };

                cv::Mat img_roi_bgr(rsrc_bgr(roi));
                cv::Mat img_roi_bgr_filt;

                // do smoothing of BGR ROI prior to color test
                cv::medianBlur(img_roi_bgr, img_roi_bgr_filt, 3);

                // equalize gray ROI
                cv::Mat img_filt_equ;
                cv::equalizeHist(img_roi, img_filt_equ);

#ifdef _COLLECT_SAMPLES
                if (samp_ct < 1000)
                {
                    cv::Mat img_samp = img_roi_bgr;
                    //cv::cvtColor(img_filt_equ, img_samp, cv::COLOR_GRAY2BGR);
                    //cv::cvtColor(img_roi_bgr_filt, img_samp, cv::COLOR_BGR2HSV);

                    int k = tmpl_gray_p.size().width + 4;
                    int x = (samp_ct % sampx) * k;
                    int y = (samp_ct / sampx) * k;
                    cv::Rect roi0 = { {x,y}, cv::Size(k,k) };
                    cv::Rect roi1 = { {x + 1, y + 1}, cv::Size(k - 2, k - 2) };
                    // surround each sample with a white border that can be manually re-colored
                    cv::rectangle(samples, roi1, { 255,255,255 });
                    cv::Rect roi2 = { {x + 2, y + 2}, cv::Size(k - 4, k - 4) };

                    img_samp.copyTo(samples(roi2));
                    samp_ct++;
#if 0
                    x = (samp_ct % sampx) * k;
                    y = (samp_ct / sampx) * k;
                    cv::Rect roi3 = { {x + 2, y + 2}, cv::Size(dct_fv.dim(), dct_fv.dim()) };
                    grel.copyTo(samples(roi3));
                    samp_ct++;
#endif
                }
#endif
                // sqdiff shape test on filtered, gray, equalized ROI
                cv::Mat tmatchx;
                cv::Mat& rtmpl = (lminfo.corr > 0.0) ? tmpl_gray_p : tmpl_gray_n;
                matchTemplate(img_filt_equ, rtmpl, tmatchx, cv::TM_SQDIFF_NORMED);
                lminfo.rmatch = tmatchx.at<float>(0, 0);
                bool is_sqdiff_test_ok = (lminfo.rmatch < thr_sqdiff);

                // optional color test
                bool is_color_test_ok = true;
                if (is_sqdiff_test_ok && is_color_id_enabled)
                {
                    identify_colors(img_roi_bgr_filt, lminfo);
                    is_color_test_ok = (lminfo.code != -1);
                }

                if (is_sqdiff_test_ok && is_color_test_ok)
                {
                    // this is a landmark
                    rinfo.push_back(lminfo);
                }
            }
        }
    }



    ///////////////////////////////////////////////////////////////////////////////
    // PUBLIC CLASS STATIC FUNCTIONS

    void BGRLandmark::create_landmark_image(
        cv::Mat& rimg,
        const double dim_grid,
        const double dim_border,
        const grid_colors_t& rcolors,
        const cv::Scalar border_color,
        const int dpi)
    {
        // set limits on 2x2 grid size (0.5 inch to 6.0 inch)
        double dim_grid_fix = apply_rail<double>(dim_grid, 0.5, 6.0);

        // set limits on size of border (0 inches to 1 inch)
        double dim_border_fix = apply_rail<double>(dim_border, 0.0, 1.0);

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

        // fill in 2x2 blocks (clockwise from upper left)
        cv::rectangle(img_grid, { 0, 0, kgridh, kgridh }, colors[0], -1);
        cv::rectangle(img_grid, { kgridh, 0, kgridh, kgridh }, colors[1], -1);
        cv::rectangle(img_grid, { kgridh, kgridh, kgridh, kgridh }, colors[2], -1);
        cv::rectangle(img_grid, { 0, kgridh, kgridh, kgridh }, colors[3], -1);

        // copy grid into image with border
        cv::Rect roi = cv::Rect({ kborder, kborder }, img_grid.size());
        img_grid.copyTo(rimg(roi));
    }



    // creates printable multi-landmark image by repeating 2x2 landmark patterns
    void BGRLandmark::create_multi_landmark_image(
        cv::Mat& rimg,
        const std::string& rslabels,
        const int xrepeat,
        const int yrepeat,
        const double dim_grid,
        const double dim_spacing,
        const double dim_border,
        const cv::Scalar border_color,
        const int dpi)
    {
        // set limits on 2x2 grid size (0.25 inch to 2.0 inch)
        double dim_grid_fix = apply_rail<double>(dim_grid, 0.25, 2.0);

        // set limits on inter-grid spacing (0.25 inch to 8.0 inch)
        double dim_spacing_fix = apply_rail<double>(dim_spacing, 0.25, 8.0);

        // set limits on size of border (0 inches to 1 inch)
        double dim_border_fix = apply_rail<double>(dim_border, 0.0, 1.0);

        const int kgrid = static_cast<int>(dim_grid_fix * dpi);
        const int kspacing = static_cast<int>(dim_spacing_fix * dpi);
        const int kborder = static_cast<int>(dim_border_fix * dpi);
        const int kborder2 = 2 * kborder;

        // set arbitrary limits on repeat counts
        int xrfix = apply_rail<int>(xrepeat, 1, 8);
        int yrfix = apply_rail<int>(yrepeat, 1, 8);

        // create image that will contain border and the multiple landmarks
        // then fill in border and white background for landmarks
        const int kbx = (((xrfix - 1) * kspacing) + kgrid) + kborder2;
        const int kby = (((yrfix - 1) * kspacing) + kgrid) + kborder2;
        rimg = cv::Mat::zeros({ kbx, kby }, CV_8UC3);
        cv::rectangle(rimg, { 0, 0, kbx, kby }, border_color, -1);
        cv::rectangle(rimg, { kborder, kborder, kbx - kborder2, kby - kborder2 }, { 255,255,255 }, -1);

        // draw the landmarks into the image
        // cycle through the label character string to pick current landmark pattern
        int k = 0;
        int kmax = static_cast<int>(rslabels.size());
        for (int j = 0; j < yrfix; j++)
        {
            for (int i = 0; i < xrfix; i++)
            {
                cv::Mat img;
                char c = (kmax) ? rslabels.at(k % kmax) : 'A';
                create_landmark_image(img, dim_grid_fix, 0.0, PATTERN_MAP.find(c)->second);
                int offseti = kborder + (i * kspacing);
                int offsetj = kborder + (j * kspacing);
                cv::Rect roi = { offseti, offsetj, kgrid, kgrid };
                img.copyTo(rimg(roi));
                k++;
            }
        }
    }



    ///////////////////////////////////////////////////////////////////////////////
    // PRIVATE CLASS STATIC FUNCTIONS

    void BGRLandmark::create_template_image(
        cv::Mat& rimg,
        const int k,
        const grid_colors_t& rcolors)
    {
        const int kh = k / 2;

        // set colors of each square in 2x2 grid, index is clockwise from upper left
        cv::Scalar colors[4];
        colors[0] = BGR_COLORS[static_cast<int>(rcolors.c00)];
        colors[1] = BGR_COLORS[static_cast<int>(rcolors.c01)];
        colors[2] = BGR_COLORS[static_cast<int>(rcolors.c11)];
        colors[3] = BGR_COLORS[static_cast<int>(rcolors.c10)];

        rimg = cv::Mat::zeros({ k, k }, CV_8UC3);

        // fill in 2x2 squares (clockwise from upper left)
        cv::rectangle(rimg, { 0, 0, kh, kh }, colors[0], -1);
        cv::rectangle(rimg, { kh + 1, 0, kh, kh }, colors[1], -1);
        cv::rectangle(rimg, { kh, kh, k - 1, k - 1 }, colors[2], -1);
        cv::rectangle(rimg, { 0, kh + 1, kh, kh }, colors[3], -1);

        // fill in average at borders between squares
        cv::Scalar avg_c00_c10 = (colors[0] + colors[1]) / 2;
        cv::Scalar avg_c10_c11 = (colors[1] + colors[2]) / 2;
        cv::Scalar avg_c11_c01 = (colors[2] + colors[3]) / 2;
        cv::Scalar avg_c01_c00 = (colors[3] + colors[0]) / 2;
        cv::line(rimg, { kh, 0 }, { kh, kh }, avg_c00_c10);
        cv::line(rimg, { kh, kh }, { k - 1, kh }, avg_c10_c11);
        cv::line(rimg, { kh, kh }, { kh, k - 1 }, avg_c11_c01);
        cv::line(rimg, { 0, kh }, { kh, kh }, avg_c01_c00);

        // central point gets average of all squares
        cv::Scalar avg_all = (colors[0] + colors[1] + colors[2] + colors[3]) / 4;
        cv::line(rimg, { kh, kh }, { kh, kh }, avg_all);
    }



    void BGRLandmark::identify_colors(const cv::Mat& rimg, BGRLandmark::landmark_info_t& rinfo) const
    {
        cv::Vec3f pc0;
        cv::Vec3f pc1;
        cv::Vec3f pg0;
        cv::Vec3f pg1;

        // sample the corners
        // locations are offset by 1 pixel in X and Y and filtering is 3x3 
        // so each sample will be 9 unique pixels smoothed together
        if (rinfo.corr > 0)
        {
            // "positive" landmark
            pg0 = rimg.at<cv::Vec3b>(1, 1);
            pg1 = rimg.at<cv::Vec3b>(kdim - 2, kdim - 2);
            pc0 = rimg.at<cv::Vec3b>(1, kdim - 2);
            pc1 = rimg.at<cv::Vec3b>(kdim - 2, 1);
        }
        else
        {
            // "negative" landmark
            pg0 = rimg.at<cv::Vec3b>(1, kdim - 2);
            pg1 = rimg.at<cv::Vec3b>(kdim - 2, 1);
            pc0 = rimg.at<cv::Vec3b>(1, 1);
            pc1 = rimg.at<cv::Vec3b>(kdim - 2, kdim - 2);
        }

        // get pixel value ranges for colored corners
        double p0max, p0min, p0rng;
        double p1max, p1min, p1rng;
        cv::minMaxLoc(pc0, &p0min, &p0max);
        cv::minMaxLoc(pc1, &p1min, &p1max);
        p0rng = p0max - p0min;
        p1rng = p1max - p1min;

        // see if there's enough range in BGR components for color classification
        if ((p0rng > thr_bgr_rng) && (p1rng > thr_bgr_rng))
        {
            // get gray level for all corners
            double pg0gray = bgr_to_gray(pg0);
            double pg1gray = bgr_to_gray(pg1);
            double pc0gray = bgr_to_gray(pc0);
            double pc1gray = bgr_to_gray(pc1);

            // sanity check to see if black corners are dark and colored corners are bright
            // one color can be brighter than the other so threshold is set at 33% of range
            double qminthr = rinfo.min + (rinfo.rng * 0.333);
            if ((pg0gray < qminthr) && (pg1gray < qminthr) &&
                (pc0gray >= qminthr) && (pc1gray >= qminthr))
            { 
                const double BGR_EPS = 1.0e-6;
                int nc0 = -1;
                int nc1 = -1;

                // normalize the BGR components for each corner
                // each component will be in range 0-1
                cv::Vec3f pc0n;
                cv::Vec3f pc1n;
                cv::normalize(pc0, pc0n, 0, 1, cv::NORM_MINMAX);
                cv::normalize(pc1, pc1n, 0, 1, cv::NORM_MINMAX);

                // classify yellow-magenta-cyan (0,1,2) for the two colored corner pixels
                // by determing which component is a "absent" or minimum (normalized to 0)
                if (pc0n[0] < BGR_EPS) nc0 = 0;
                if (pc0n[1] < BGR_EPS) nc0 = 1;
                if (pc0n[2] < BGR_EPS) nc0 = 2;
                if (pc1n[0] < BGR_EPS) nc1 = 0;
                if (pc1n[1] < BGR_EPS) nc1 = 1;
                if (pc1n[2] < BGR_EPS) nc1 = 2;
                rinfo.code = get_bgr_code(rinfo.corr, nc0, nc1);
            }
        }
    }



    void BGRLandmark::identify_colors_thr(const cv::Mat& rimg, BGRLandmark::landmark_info_t& rinfo) const
    {
        cv::Vec3b pc0;
        cv::Vec3b pc1;
        cv::Vec3b pg0;
        cv::Vec3b pg1;

        // convert b001,b010,b100 -> 0,1,2
        const int bitcode[5] = { -1, 0, 1, -1, 2 };
        uint8_t xx = 2;

        // 80-150 original S

        // HSV yellow (12-25)
        cv::Vec3b vlo0 = { 3, 70, 0 };
        cv::Vec3b vhi0 = { 34, 160, 255 };
        
        // HSV magenta (154-170)
        cv::Vec3b vlo1 = { 145, 70, 0 };
        cv::Vec3b vhi1 = { 179, 160, 255 };
        
        // HSV cyan (96-110)
        cv::Vec3b vlo2 = { 87, 70, 0 };
        cv::Vec3b vhi2 = { 119, 160, 255 };

        cv::Mat ximg;
        cv::cvtColor(rimg, ximg, cv::COLOR_BGR2HSV);

        // sample the corners
        // locations are offset by 1 pixel in X and Y and filtering is 3x3 
        // so each sample will be 9 unique pixels smoothed together
        if (rinfo.corr > 0)
        {
            // "positive" landmark
            pg0 = ximg.at<cv::Vec3b>(1, 1);
            pg1 = ximg.at<cv::Vec3b>(kdim - 2, kdim - 2);
            pc0 = ximg.at<cv::Vec3b>(1, kdim - 2);
            pc1 = ximg.at<cv::Vec3b>(kdim - 2, 1);
        }
        else
        {
            // "negative" landmark
            pg0 = ximg.at<cv::Vec3b>(1, kdim - 2);
            pg1 = ximg.at<cv::Vec3b>(kdim - 2, 1);
            pc0 = ximg.at<cv::Vec3b>(1, 1);
            pc1 = ximg.at<cv::Vec3b>(kdim - 2, kdim - 2);
        }

        cv::Mat x00, x01, x02;
        cv::Mat x10, x11, x12;
        
        cv::inRange(pc0, vlo0, vhi0, x00);
        cv::inRange(pc0, vlo1, vhi1, x01);
        cv::inRange(pc0, vlo2, vhi2, x02);
        cv::inRange(pc1, vlo0, vhi0, x10);
        cv::inRange(pc1, vlo1, vhi1, x11);
        cv::inRange(pc1, vlo2, vhi2, x12);
        int a = (x00.at<uint8_t>(0, 0) & 1) |
                ((x01.at<uint8_t>(0, 0) & 1) << 1) |
                ((x02.at<uint8_t>(0, 0) & 1) << 2);
        int b = (x10.at<uint8_t>(0, 0) & 1) |
            ((x11.at<uint8_t>(0, 0) & 1) << 1) |
            ((x12.at<uint8_t>(0, 0) & 1) << 2);
        rinfo.code = get_bgr_code(rinfo.corr, bitcode[a], bitcode[b]);
    }



    int BGRLandmark::get_bgr_code(double s, const int a, const int b)
    {
        // a and b must be in range [0,1,2] and must not be equal
        // otherwise the code conversion won't work
        // positive pattern is code 0-5, negative pattern is code 6-11
        int code = -1;
        if ((a != b) && (a >= 0) && (a <= 2) && (b >= 0) && (b <= 2))
        {
            if (a == 0)
            {
                code = (b == 1) ? 0 : 1;  // 0,1 or 0,2
            }
            else if (a == 1)
            {
                code = (b == 0) ? 2 : 3;  // 1,0 or 1,2
            }
            else
            {
                code = (b == 0) ? 4 : 5;  // 2,0 or 2,1
            }
            if (s < 0.0) code += 6; // negative pattern
        }
        return code;
    }
}

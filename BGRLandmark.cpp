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
        cv::Scalar(0, 255, 255),    // yellow
        cv::Scalar(255, 0, 0),      // blue
        cv::Scalar(255, 0, 255),    // magenta
        cv::Scalar(255, 255, 0),    // cyan
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



    // returns a value "railed" to fall within a max-min range
    template <class T>
    static T apply_rail(const T v, const T vmin, const T vmax)
    {
        return (v > vmax) ? vmax : ((v < vmin) ? vmin : v);
    }



    BGRLandmark::BGRLandmark() :
        dct_fv(4, 1, 9)
    {
        init();
#ifdef _DEBUG
        for (int i = 0; i < 12; i++)
        {
            cv::Mat img1;
            char c = i + 'A';
            std::string s = "dbg_bgrlm_" + std::string(1, c) + ".png";
            create_landmark_image(img1, 3.0, 0.25, PATTERN_MAP.find(c)->second, { 255,255,255 });
            cv::imwrite(s, img1);
        }
        cv::Mat img2;
        create_multi_landmark_image(img2, "ADGJBEHKCFIL", 4, 3, 0.5, 2.25, 0.25, { 192,192,192 });
        cv::imwrite("dbg_multi.png", img2);
        create_multi_landmark_image(img2, "AG", 2, 1, 0.5, 8, 0.0);
        cv::imwrite("dbg_double.png", img2);
#endif
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
        const int thr_bgr_rng)
    {
        // fix k to be odd and in range 9-15
        int fixk = ((k / 2) * 2) + 1;
        kdim = apply_rail<int>(fixk, 9, 15);

        // apply thresholds
        // TODO -- assert type is CV_8U somewhere during match
        this->thr_corr = thr_corr;
        this->thr_pix_rng = thr_pix_rng;
        this->thr_pix_min = thr_pix_min;
        this->thr_bgr_rng = thr_bgr_rng;

        // create the B&W matching templates
        const grid_colors_t colors = PATTERN_MAP.find('0')->second;
        cv::Mat tmpl_bgr;
        create_template_image(tmpl_bgr, kdim, colors);
        cv::cvtColor(tmpl_bgr, tmpl_gray_p, cv::COLOR_BGR2GRAY);
        cv::rotate(tmpl_gray_p, tmpl_gray_n, cv::ROTATE_90_CLOCKWISE);

#ifdef _DEBUG
        imwrite("dbg_tmpl_gray_p.png", tmpl_gray_p);
        imwrite("dbg_tmpl_gray_n.png", tmpl_gray_n);
#endif

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

        // match the positive and negative templates
        // and find absolute difference between the two results
        cv::Mat tmatch0;
        cv::Mat tmatch1;
        matchTemplate(rsrc, tmpl_gray_p, tmatch0, xmode);
        matchTemplate(rsrc, tmpl_gray_n, tmatch1, xmode);
        rtmatch = abs(tmatch0 - tmatch1);

        // find local maxima in the match results...
        cv::Mat maxima_mask;
        cv::dilate(rtmatch, maxima_mask, cv::Mat());
        cv::compare(rtmatch, maxima_mask, maxima_mask, cv::CMP_GE);

        // then apply absolute threshold to get the best local maxima
        std::vector<std::vector<cv::Point>> contours;
        cv::Mat match_masked = (rtmatch > thr_corr);
        maxima_mask = maxima_mask & match_masked;

        // collect point locations of all local maxima
        std::vector<cv::Point> vec_maxima_pts;
        cv::findNonZero(maxima_mask, vec_maxima_pts);

        // check each maxima...
        for (const auto& rpt : vec_maxima_pts)
        {
            // positive diff means black in upper-left/lower-right
            // negative diff means black in lower-left/upper-right
            float pix_p = tmatch0.at<float>(rpt);
            float pix_n = tmatch1.at<float>(rpt);
            float diff = pix_p - pix_n;

            // extract region of interest
            const cv::Rect roi = cv::Rect(rpt, tmpl_gray_p.size());
            cv::Mat img_roi(rsrc(roi));

            // get pixel range stats in ROI
            double min_roi;
            double max_roi;
            cv::minMaxLoc(img_roi, &min_roi, &max_roi);
            double rng_roi = max_roi - min_roi;

            // a landmark ROI should have two dark squares and and two light squares
            // see if ROI has large range in pixel values and a minimum that is sufficiently dark
            if ((rng_roi > thr_pix_rng) && (min_roi < thr_pix_min))
            {
                // start filling in landmark info
                landmark_info_t lminfo{ rpt + tmpl_offset, diff, rng_roi, min_roi, -1 };

                cv::Mat img_roi_bgr(rsrc_bgr(roi));
                cv::Mat img_roi_gray(rsrc(roi));

#ifdef _COLLECT_SAMPLES
                if (samp_ct < 1000)
                {
                    cv::Mat frel;
                    cv::Mat grel;
                    dct_fv.pattern_to_dct_8U(img_roi_gray, frel);
                    cv::cvtColor(frel, grel, cv::COLOR_GRAY2BGR);

                    int k = tmpl_gray_p.size().width + 4;
                    int x = (samp_ct % sampx) * k;
                    int y = (samp_ct / sampx) * k;
                    cv::Rect roi0 = { {x,y}, cv::Size(k,k) };
                    cv::Rect roi1 = { {x + 1, y + 1}, cv::Size(k - 2, k - 2) };
                    // surround each sample with a white border that can be manually re-colored
                    cv::rectangle(samples, roi1, { 255,255,255 });
                    cv::Rect roi2 = { {x + 2, y + 2}, cv::Size(k - 4, k - 4) };
                    //cv::Mat img_roi_bgr_proc;
                    //cv::bilateralFilter(img_roi_bgr, img_roi_bgr_proc, 3, 200, 200);

                    img_roi_bgr.copyTo(samples(roi2));
                    samp_ct++;
                    x = (samp_ct % sampx) * k;
                    y = (samp_ct / sampx) * k;
                    cv::Rect roi3 = { {x + 2, y + 2}, cv::Size(dct_fv.dim(), dct_fv.dim()) };
                    grel.copyTo(samples(roi3));
                    samp_ct++;
                }
#endif

                // TODO -- maybe add some kind of additional shape test

                if (is_color_id_enabled)
                {
                    // use bilateral filter to suppress as much noise as possible in ROI
                    // while also preserving edges between colored regions
                    cv::Mat img_roi_bgr_proc;
                    cv::bilateralFilter(img_roi_bgr, img_roi_bgr_proc, 3, 200, 200);
                    identify_colors(img_roi_bgr_proc, lminfo);

                    // save it if color test gave a sane result
                    if (lminfo.code != -1)
                    {
                        rinfo.push_back(lminfo);
                    }
                }
                else
                {
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
        int kmax = rslabels.size();
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

#ifdef _DEBUG
        cv::imwrite("dbg_tmpl_bgr.png", rimg);
#endif
    }



    void BGRLandmark::identify_colors(const cv::Mat& rimg, BGRLandmark::landmark_info_t& rinfo) const
    {
        int result = -1;
        const cv::Vec3f norm_ycm[3] =
        {
            {0, 1, 1},  // 0,G,R yellow
            {1, 0, 1},  // B,0,R magenta
            {1, 1, 0},  // B,G,0 cyan 
        };

        cv::Vec3f p0;
        cv::Vec3f p1;

        // find BGR at appropriate colored corners
        if (rinfo.diff > 0)
        {
            // "positive" landmark
            p0 = rimg.at<cv::Vec3b>(0, kdim - 1);
            p1 = rimg.at<cv::Vec3b>(kdim - 1, 0);
        }
        else
        {
            // "negative" landmark
            p0 = rimg.at<cv::Vec3b>(0, 0);
            p1 = rimg.at<cv::Vec3b>(kdim - 1, kdim - 1);
        }

        // get pixel value ranges for corners
        double p0max, p0min, p0rng;
        double p1max, p1min, p1rng;
        cv::minMaxLoc(p0, &p0min, &p0max);
        cv::minMaxLoc(p1, &p1min, &p1max);
        p0rng = p0max - p0min;
        p1rng = p1max - p1min;

        // then normalize the BGR components for each corner
        // each value will fall in range 0-1
        cv::normalize(p0, p0, 0, 1, cv::NORM_MINMAX);
        cv::normalize(p1, p1, 0, 1, cv::NORM_MINMAX);

        // this BGR "score" (sum of all components) should range from 1 to 2
        // something in the middle means a yellow-magenta-cyan match can be performed
        double s0 = p0[0] + p0[1] + p0[2];
        double s1 = p1[0] + p1[1] + p1[2];

        // see if there's enough range in BGR components
        // for a valid yellow-magenta-cyan classification
        if ((p0rng > thr_bgr_rng) && (p1rng > thr_bgr_rng))
        {
            const double BGR_EPS = 1.0e-6;
            int nc0 = -1;
            int nc1 = -1;
            // classify yellow-magenta-cyan for the two colored corner pixels
            // by determing which component is a "absent" or minimum (normalized to 0)
            if (p0[0] < BGR_EPS) nc0 = 0;
            if (p0[1] < BGR_EPS) nc0 = 1;
            if (p0[2] < BGR_EPS) nc0 = 2;
            if (p1[0] < BGR_EPS) nc1 = 0;
            if (p1[1] < BGR_EPS) nc1 = 1;
            if (p1[2] < BGR_EPS) nc1 = 2;
            rinfo.code = get_bgr_code(rinfo.diff, nc0, nc1);
        }
    }



    int BGRLandmark::get_bgr_code(double s, const int a, const int b)
    {
        // a and b must be in range [0,1,2] and must not be equal
        // otherwise the code conversion won't work
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
            if (s < 0.0) code += 6;
        }
        return code;
    }
}

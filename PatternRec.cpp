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

#include <fstream>
#include <algorithm>
#include "opencv2/highgui.hpp"
#include "BGRLandmark.h"
#include "PatternRec.h"



PatternRec::PatternRec()
{
    // generate DCT zigzag point lookup vector
    get_zigzag_pts(kdct, _vzzpts);
}



PatternRec::~PatternRec()
{

}



bool PatternRec::load_pca(const std::string& rs, cv::PCA& rpca)
{
    bool is_ok = false;
    cv::FileStorage cvfs;
    cvfs.open(rs, cv::FileStorage::READ);
    if (cvfs.isOpened())
    {
        cv::FileNode cvfn = cvfs.root();
        rpca.read(cvfn);
        cvfs.release();
        is_ok = true;
    }
    return is_ok;
}



bool PatternRec::run_csv_to_pca(
    const std::string& rsin,
    const std::string& rsout,
    const double var_keep_fac)
{
    bool is_ok = false;

    cv::Mat img_pca;
    if (read_csv_into_mat(rsin, img_pca))
    {
        cv::PCA mypca(img_pca, cv::noArray(), cv::PCA::DATA_AS_ROW, var_keep_fac);
        cv::FileStorage cvfs;
        cvfs.open(rsout, cv::FileStorage::WRITE);
        if (cvfs.isOpened())
        {
            mypca.write(cvfs);
            cvfs.release();
            is_ok = true;
        }
    }
    return is_ok;
}



bool PatternRec::read_csv_into_mat(const std::string& rs, cv::Mat& rimg)
{
    bool is_ok = true;
    std::ifstream ifs;
    std::vector<cv::Mat> vv;
    size_t vcols = 0;

    rimg.release();

    ifs.open(rs.c_str());
    if (ifs.is_open())
    {
        std::string s;
        while (std::getline(ifs, s) && is_ok)
        {
            cv::Mat one_row;
            std::replace(s.begin(), s.end(), ',', ' ');
            std::istringstream iss(s);
            while (!iss.eof() && is_ok)
            {
                float val;
                iss >> val;
                one_row.push_back(val);
            }
            
            // do a sanity check for matching vector size after first vector is read
            one_row = one_row.t();
            if (vcols == 0)
            {
                vcols = one_row.cols;
            }
            else if (one_row.cols != vcols)
            {
                is_ok = false;
            }
            
            rimg.push_back(one_row);
        }
        ifs.close();
    }

    return is_ok;
}



void PatternRec::spew_float_vecs_to_csv(
    const std::string& rs,
    const std::string& rsuffix,
    std::vector<std::vector<float>>& rvv)
{
    std::ofstream ofs;
    std::string sname = rs + rsuffix + ".csv";
    ofs.open(sname.c_str());
    if (ofs.is_open())
    {
        for (const auto& rv : rvv)
        {
            bool is_first = true;
            for (const auto& rf : rv)
            {
                if (!is_first)
                {
                    ofs << ",";
                }
                ofs << rf;
                is_first = false;
            }
            ofs << std::endl;
        }
        ofs.close();
    }
}



void PatternRec::get_zigzag_pts(const int k, std::vector<cv::Point>& rvec)
{
    cv::Point pt = { 0,0 };
    enum edir { EAST, SW, SOUTH, NE };
    enum edir zdir = EAST;
    int nn = k * k;
    int kstop = k - 1;
    for (int i = 0; i < nn; i++)
    {
        rvec.push_back(pt);
        if (zdir == EAST)
        {
            pt.x++;
            zdir = (pt.y == kstop) ? NE : SW;
        }
        else if (zdir == SW)
        {
            pt.x--;
            pt.y++;
            if (pt.y == kstop)
            {
                zdir = EAST;
            }
            else if (pt.x == 0)
            {
                zdir = SOUTH;
            }
        }
        else if (zdir == SOUTH)
        {
            pt.y++;
            zdir = (pt.x == kstop) ? SW : NE;
        }
        else if (zdir == NE)
        {
            pt.x++;
            pt.y--;
            if (pt.x == kstop)
            {
                zdir = SOUTH;
            }
            else if (pt.y == 0)
            {
                zdir = EAST;
            }
        }
    }
}



bool PatternRec::load_samples_from_img(
    const std::string& rsfile,
    const int maxsampct,
    const bool is_axes_flipped)
{
    const int kdctcompct = kdctmaxcomp - kdctmincomp + 1;

    cv::Mat img_gray;
    cv::Mat img = cv::imread(rsfile, cv::IMREAD_COLOR);

    if (img.size() == cv::Size{0, 0})
    {
        return false;
    }

    if (is_axes_flipped)
    {
        cv::flip(img, img, -1);
    }

    cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);

    // determine sizes of everything from known samples count
    cv::Size sz = img.size();
    cv::Size sz_box = cv::Size(sz.width / SAMP_NUM_X, sz.height / SAMP_NUM_Y);
    cv::Size sz_roi = cv::Size(sz_box.width - 4, sz_box.height - 4);
    cv::Point ctr{ sz_roi.width / 2, sz_roi.height / 2 };
    kdim = sz_roi.width;
    krad = kdim / 2;

    // now that dimension is known a BGRLandmark matcher can be created
    BGRLandmark bgrm;
    bgrm.init(kdim);
    bgrm.set_color_id_enable(false);

    // temporary vectors for the data...
    std::vector<std::vector<float>> vvp;
    std::vector<std::vector<float>> vvn;
    std::vector<std::vector<float>> vv0;

    // loop through all the sample images...
    for (int j = 0; j < SAMP_NUM_Y; j++)
    {
        for (int i = 0; i < SAMP_NUM_X; i++)
        {
            // get offset for box, rectangle border, and ROI
            cv::Point pt0{ i * sz_box.width, j * sz_box.height };
            cv::Point pt1 = pt0 + cv::Point{ 1, 1 };
            cv::Point pt2 = pt1 + cv::Point{ 1, 1 };

            // get gray ROI and BGR pixel at corner of border rectangle in source image
            cv::Rect roi(pt2, sz_roi);
            cv::Mat img_roi = img_gray(roi);
            cv::Scalar bgr_pixel = img.at<cv::Vec3b>(pt1);

            // these should all match because they were captured with same settings
            cv::Mat img_match;
            std::vector<BGRLandmark::landmark_info_t> lminfo;
            bgrm.perform_match(img(roi), img_roi, img_match, lminfo);

            // make a square image of fixed dim
            // convert to float in -128 to 127 range
            // then run DCT on it (just like a JPEG block)
            cv::Mat img_dct, img_src, img_src_32f;
            cv::resize(img_roi, img_src, { kdct,kdct });
            img_src.convertTo(img_src_32f, CV_32F);
            img_src_32f -= 128.0;
            cv::dct(img_src_32f, img_dct, 0);

            // create a feature vector for this sample
            // by extracting components from the DCT using zigzag traversal
            std::vector<float> vfeature;
            for (int ii = kdctmincomp; ii <= kdctmaxcomp; ii++)
            {
                float val = img_dct.at<float>(_vzzpts[ii]);
                vfeature.push_back(val);
            }

            // stick feature vector in appropriate structure
            if ((bgr_pixel != cv::Scalar{ 255, 255, 255 }) || (lminfo.size() == 0))
            {
                // non-white border (or no match from BGRLandmark) is a junk sample
                // a "red" border indicates junk but the red in the images has goofy BGR values
                vv0.push_back(vfeature);
            }
            else if (lminfo[0].diff < 0)
            {
                // this is a "negative" sample
                vvn.push_back(vfeature);
            }
            else
            {
                // this is a "positive" sample
                vvp.push_back(vfeature);
            }
        }
    }

    // the samples have a crude ordering based on how they were collected
    // so shuffle in case we don't want to use all the samples
    // this insures a subset has similar variation
    if (maxsampct > 0)
    {
        std::random_shuffle(vvp.begin(), vvp.end());
        std::random_shuffle(vvn.begin(), vvn.end());
        std::random_shuffle(vv0.begin(), vv0.end());
        if (vvp.size() > maxsampct) vvp.resize(maxsampct);
        if (vvn.size() > maxsampct) vvn.resize(maxsampct);
        if (vv0.size() > maxsampct) vv0.resize(maxsampct);
    }

    // accumulate the data
    _vvp.insert(_vvp.end(), vvp.begin(), vvp.end());
    _vvn.insert(_vvn.end(), vvn.begin(), vvn.end());
    _vv0.insert(_vv0.end(), vv0.begin(), vv0.end());

    return true;
}



void PatternRec::save_samples_to_csv(const std::string& rsprefix)
{
    spew_float_vecs_to_csv(rsprefix, "_p", _vvp);
    spew_float_vecs_to_csv(rsprefix, "_n", _vvn);
    spew_float_vecs_to_csv(rsprefix, "_0", _vv0);
}



void PatternRec::samp_to_pattern(const std::vector<float>& rsamp, cv::Mat& rimg)
{
    cv::Mat img_dct = cv::Mat::zeros({ kdct, kdct }, CV_32F);
    
    // reconstruct DCT components
    int mm = 0;
    for (int ii = kdctmincomp; ii <= kdctmaxcomp; ii++)
    {
        img_dct.at<float>(_vzzpts[ii]) = rsamp[mm];
        mm++;
    }

    // invert DCT and rescale for a gray image
    cv::Mat img_idct;
    cv::idct(img_dct, img_idct, 0);
    cv::normalize(img_idct, img_idct, 0, 255, cv::NORM_MINMAX);
    img_idct.convertTo(rimg, CV_8U);
}

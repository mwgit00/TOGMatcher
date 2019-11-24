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



PatternRec::PatternRec() : dct_fv(8, 1, 9)
{
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
                double val;
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



void PatternRec::spew_double_vecs_to_csv(
    const std::string& rs,
    const std::string& rsuffix,
    std::vector<std::vector<double>>& rvv)
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



bool PatternRec::load_samples_from_img(
    const std::string& rsfile,
    const int maxsampct,
    const bool is_horiz_flipped)
{
    const int kdctcompct = dct_fv.fvsize();

    cv::Mat img_gray;
    cv::Mat img = cv::imread(rsfile, cv::IMREAD_COLOR);

    if (img.size() == cv::Size{0, 0})
    {
        return false;
    }

    if (is_horiz_flipped)
    {
        cv::flip(img, img, 0);
    }

    cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);

    // determine sizes of everything from known samples count
    cv::Size sz = img.size();
    cv::Size sz_box = cv::Size(sz.width / SAMP_NUM_X, sz.height / SAMP_NUM_Y);
    cv::Size sz_roi = cv::Size(sz_box.width - 4, sz_box.height - 4);
    cv::Point ctr{ sz_roi.width / 2, sz_roi.height / 2 };
    kdim = sz_roi.width;

    // now that dimension is known a BGRLandmark matcher can be created
    cpoz::BGRLandmark bgrm;
    bgrm.init(kdim);
    bgrm.set_color_id_enable(false);

    // temporary vectors for the data...
    std::vector<std::vector<double>> vvp;
    std::vector<std::vector<double>> vvn;
    std::vector<std::vector<double>> vv0;

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
            std::vector<cpoz::BGRLandmark::landmark_info_t> lminfo;
            bgrm.perform_match(img(roi), img_roi, img_match, lminfo);

            std::vector<double> vfeature;
            dct_fv.pattern_to_features(img_roi, vfeature);

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
    // this insures a subset has similar variation (maybe)
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
    spew_double_vecs_to_csv(rsprefix, "_p", _vvp);
    spew_double_vecs_to_csv(rsprefix, "_n", _vvn);
    spew_double_vecs_to_csv(rsprefix, "_0", _vv0);
}

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

#include "opencv2/highgui.hpp"
#include "DCTFeature.h"



DCTFeature::DCTFeature(const int k, const int imin, const int imax)
{
    init(k, imin, imax);
}



DCTFeature::~DCTFeature()
{
    // does nothing
}



void DCTFeature::init(const int k, const int imin, const int imax)
{
    kdim = k;
    kmincomp = imin;
    kmaxcomp = imax;
    
    // generate DCT zigzag point lookup vector (same as what JPEG does)
    generate_zigzag_pts(kdim, vzigzagpts);

    // stash size of feature vector
    kfvsize = (kmaxcomp - kmincomp) + 1;

    // consider the current mean and inv. covariance matrices to be invalid
    is_stats_loaded = false;
}


bool DCTFeature::load(const std::string& rs)
{
    is_stats_loaded = false;
    try
    {
        int k, imin, imax;
        cv::FileStorage cvfs;
        
        cvfs.open(rs, cv::FileStorage::READ);
        cvfs["dct_kdim"] >> k;
        cvfs["dct_kmincomp"] >> imin;
        cvfs["dct_kmaxcomp"] >> imax;

        vstats.clear();
        cv::FileNode nodem = cvfs["stats"];
        for (cv::FileNodeIterator it = nodem.begin(); it != nodem.end(); it++)
        {
            cv::FileNode item = *it;
            vstats.emplace_back(T_STATS());
            T_STATS& rx = vstats.back();
            item["name"] >> rx.name;
            item["mean"] >> rx.mean;
            item["invcov"] >> rx.invcov;
            item["thr"] >> rx.thr;
            rx.is_loaded = true;
        }

        init(k, imin, imax);
        is_stats_loaded = true;
    }
    catch (std::exception& ex)
    {
        vstats.clear();
    }
    return is_stats_loaded;
}



double DCTFeature::dist(const size_t idx, const std::vector<double>& rfv) const
{
    double r = DBL_MAX;
    if (idx < vstats.size())
    {
        r = cv::Mahalanobis(rfv, vstats[idx].mean, vstats[idx].invcov);
    }
    return r;
}

bool DCTFeature::is_match(
    const size_t idx,
    const std::vector<double>& rfv,
    double* pdist) const
{
    bool result = false;
    if (idx < vstats.size())
    {
        double r = cv::Mahalanobis(rfv, vstats[idx].mean, vstats[idx].invcov);
        if (pdist != nullptr) *pdist = r;
        result = (r < vstats[idx].thr);
    }
    return result;
}



void DCTFeature::pattern_to_dct_64F(const cv::Mat& rimg, cv::Mat& rdct64F) const
{
    // shrink input to square image of size for DCT
    // convert to double in -128 to 127 range
    // then run DCT on it (just like a JPEG block)
    cv::Mat img_src;
    cv::Mat img_src_64F;
    cv::resize(rimg, img_src, cv::Size(kdim, kdim), 0, 0, cv::INTER_AREA);
    img_src.convertTo(img_src_64F, CV_64F);
    img_src_64F -= 128.0;
    cv::dct(img_src_64F, rdct64F, 0);
}



void DCTFeature::pattern_to_dct_8U(const cv::Mat& rimg, cv::Mat& rdct8U) const
{
    cv::Mat img_dct;
    pattern_to_dct_64F(rimg, img_dct);
    cv::normalize(img_dct, img_dct, 0, 255, cv::NORM_MINMAX);
    img_dct.convertTo(rdct8U, CV_8U);
}



void DCTFeature::pattern_to_features(const cv::Mat& rimg, std::vector<double>& rfv) const
{
    cv::Mat img_dct;
    pattern_to_dct_64F(rimg, img_dct);

    // extract desired components from DCT to get feature vector
    rfv.resize(kfvsize);
    size_t mm = 0;
    for (size_t ii = kmincomp; ii <= kmaxcomp; ii++)
    {
        rfv[mm] = img_dct.at<double>(vzigzagpts[ii]);
        mm++;
    }
}



void DCTFeature::features_to_pattern(const std::vector<double>& rfv, cv::Mat& rimg) const
{
    cv::Mat img_dct = cv::Mat::zeros({ kdim, kdim }, CV_64F);

    // reconstruct DCT components
    size_t mm = 0;
    for (size_t ii = kmincomp; ii <= kmaxcomp; ii++)
    {
        if (mm < rfv.size())
        {
            img_dct.at<double>(vzigzagpts[ii]) = rfv[mm];
        }
        mm++;
    }

    // invert DCT and rescale for a gray image
    cv::Mat img_idct;
    cv::idct(img_dct, img_idct, 0);
    cv::normalize(img_idct, img_idct, 0, 255, cv::NORM_MINMAX);
    img_idct.convertTo(rimg, CV_8U);
}



void DCTFeature::generate_zigzag_pts(const int k, std::vector<cv::Point>& rvec)
{
    cv::Point pt = { 0, 0 };
    enum edir { EAST, SW, SOUTH, NE };
    enum edir zdir = EAST;
    int nn = k * k;
    int kstop = k - 1;

    rvec.clear();
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

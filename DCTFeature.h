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

#ifndef DCT_FEATURE_H_
#define DCT_FEATURE_H_

#include <vector>
#include "opencv2/imgproc.hpp"


class DCTFeature
{
public:

    typedef struct _T_STATS_struct
    {
        cv::Mat mean;       // mean vector for data
        cv::Mat invcov;     // inverse covariance matrix for data
        double thr;         // threshold for Mahalanobis distance match
        std::string name;   // name for this feature
        bool is_loaded;     // true if this record has valid data
    } T_STATS;

    // default DCT is run on 8x8 image and components 1-20 are used
    // (components 1-20 correspond to first 5 "zig-zags")
    
    DCTFeature(const int k = 8, const int imin = 1, const int imax = 20);
    virtual ~DCTFeature();

    void init(const int k, const int imin, const int imax);
    bool load(const std::string& rs);
    
    bool is_loaded() const { return is_stats_loaded; }
    int dim(void) const { return kdim; }
    int imin(void) const { return kmincomp; }
    int imax(void) const { return kmaxcomp; }
    size_t fvsize(void) const { return kfvsize; }

    double dist(const size_t idx, const std::vector<double>& rfv) const;
    
    bool is_match(
        const size_t idx,
        const std::vector<double>& rfv,
        double * pdist = nullptr) const;

    // convert 2D pattern to DCT (64-bit floating point)
    void pattern_to_dct_64F(const cv::Mat& rimg, cv::Mat& rdct64F) const;

    // convert 2D pattern to DCT (8-bit unsigned)
    void pattern_to_dct_8U(const cv::Mat& rimg, cv::Mat& rdct8U) const;

    // convert 2D pattern into a feature vector
    void pattern_to_features(const cv::Mat& rimg, std::vector<double>& rfv) const;

    // reconstruct 2D pattern from a feature vector
    void features_to_pattern(const std::vector<double>& rfv, cv::Mat& rimg) const;

    const std::vector<cv::Point>& get_zigzag_pts(void) const { return vzigzagpts; }

public:

    // generates a vector of points that traverses
    // a "zig-zag" path through a square matrix
    // this mimics how a JPEG block is encoded
    static void generate_zigzag_pts(const int k, std::vector<cv::Point>& rvec);

private:

    int kdim;
    int kmincomp;
    int kmaxcomp;
    size_t kfvsize;

    std::vector<cv::Point> vzigzagpts;

    std::vector<T_STATS> vstats;
    bool is_stats_loaded;
};

#endif // DCT_FEATURE_H_

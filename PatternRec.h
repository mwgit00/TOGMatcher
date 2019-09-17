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

#ifndef PATTERN_REC_H_
#define PATTERN_REC_H_

#include <string>
#include "opencv2/imgproc.hpp"


class PatternRec
{
public:

    // hard-coded for the sample images
    const int SAMP_NUM_X = 40;
    const int SAMP_NUM_Y = 25;

    PatternRec();
    virtual ~PatternRec();

    void clear() { _vvp.clear(); _vvn.clear(); _vv0.clear(); }

    const std::vector<float>& get_p_sample(const int i) { return _vvp[i]; }
    const std::vector<float>& get_n_sample(const int i) { return _vvn[i]; }
    const std::vector<float>& get_0_sample(const int i) { return _vv0[i]; }
    
    bool load_samples_from_img(
        const std::string& rsfile,
        const int maxsampct = -1,
        const bool is_axes_flipped = false);

    void save_samples_to_csv(const std::string& rsprefix);

    void samp_to_pattern(const std::vector<float>& rsamp, cv::Mat& rimg);

public:

    static bool load_pca(const std::string& rs, cv::PCA& rpca);

    static bool run_csv_to_pca(
        const std::string& rsin,
        const std::string& rsout,
        const double var_keep_fac);

    static bool read_csv_into_mat(const std::string& rs, cv::Mat& rimg);
    
    static void spew_float_vecs_to_csv(
        const std::string& rs,
        const std::string& rsuffix,
        std::vector<std::vector<float>>& rvv);
    
    // generates a vector of points that traverses
    // a "zig-zag" path through a square matrix
    // this mimics how a JPEG block is encoded
    static void get_zigzag_pts(const int k, std::vector<cv::Point>& rvec);

private:

    int kdim;
    int krad;

    // DCT is run on 8x8 image and components 1-20 are used
    const int kdct = 8;
    const int kdctmincomp = 1;
    const int kdctmaxcomp = 20;

    std::vector<std::vector<float>> _vvp;
    std::vector<std::vector<float>> _vvn;
    std::vector<std::vector<float>> _vv0;

    std::vector<cv::Point> _vzzpts;
};

#endif // PATTERN_REC_H_

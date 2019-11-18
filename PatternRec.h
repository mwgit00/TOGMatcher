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
#include "DCTFeature.h"


class PatternRec
{
public:

    // hard-coded for the sample images
    const int SAMP_NUM_X = 40;
    const int SAMP_NUM_Y = 25;

    PatternRec();
    virtual ~PatternRec();

    DCTFeature& get_dct_fv(void) { return dct_fv; }

    void clear() { _vvp.clear(); _vvn.clear(); _vv0.clear(); }

    const std::vector<double>& get_p_sample(const int i) { return _vvp[i]; }
    const std::vector<double>& get_n_sample(const int i) { return _vvn[i]; }
    const std::vector<double>& get_0_sample(const int i) { return _vv0[i]; }
    
    bool load_samples_from_img(
        const std::string& rsfile,
        const int maxsampct = -1,
        const bool is_horiz_flipped = false);

    void save_samples_to_csv(const std::string& rsprefix);

public:

    static bool load_pca(const std::string& rs, cv::PCA& rpca);

    static bool run_csv_to_pca(
        const std::string& rsin,
        const std::string& rsout,
        const double var_keep_fac);

    static bool read_csv_into_mat(const std::string& rs, cv::Mat& rimg);
    
    static void spew_double_vecs_to_csv(
        const std::string& rs,
        const std::string& rsuffix,
        std::vector<std::vector<double>>& rvv);
    
private:

    int kdim;

    std::vector<std::vector<double>> _vvp;
    std::vector<std::vector<double>> _vvn;
    std::vector<std::vector<double>> _vv0;

    DCTFeature dct_fv;
};

#endif // PATTERN_REC_H_

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

    bool convert_samples_to_csv(const std::string& rsfile, const std::string& rstag);

public:

    static bool read_csv_into_mat(const std::string& rs, cv::Mat& rimg);
    
    static void spew_float_vecs_to_csv(
        const std::string& rs,
        const std::string& rsuffix,
        std::vector<std::vector<float>>& rvv);
    
    static void get_zigzag_pts(const int k, std::vector<cv::Point>& rvec);

private:
    int kdim;
    int krad;
};

#endif // PATTERN_REC_H_

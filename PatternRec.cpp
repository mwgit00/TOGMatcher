
#include <fstream>
#include <algorithm>
#include "opencv2/highgui.hpp"
#include "BGRLandmark.h"
#include "PatternRec.h"

PatternRec::PatternRec()
{

}

PatternRec::~PatternRec()
{

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
            while (!iss.eof())
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



bool PatternRec::convert_samples_to_csv(const std::string& rsfile, const std::string& rstag)
{
    bool result = false;

    const int kdct = 8;
    const int kdctmincomp = 1;
    const int kdctmaxcomp = 20;
    const int kdctcompct = kdctmaxcomp - kdctmincomp + 1;

    cv::Mat img_gray;
    cv::Mat img = cv::imread(rsfile, cv::IMREAD_COLOR);
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

    // generate DCT zigzag point lookup vector
    std::vector<cv::Point> vzigzagPts;
    get_zigzag_pts(kdct, vzigzagPts);

    // create training data structures
    std::vector<std::vector<float>> vdatp;
    std::vector<std::vector<float>> vdatn;
    std::vector<std::vector<float>> vdat0;
    
    // loop through all the sample images...
    for (int j = 0; j < SAMP_NUM_Y; j++)
    {
        for (int i = 0; i < SAMP_NUM_X; i++)
        {
            // get offset for box, rectangle border, and ROI
            cv::Point pt0{ i * sz_box.width, j * sz_box.height };
            cv::Point pt1 = pt0 + cv::Point{ 1, 1};
            cv::Point pt2 = pt1 + cv::Point{ 1, 1 };
            
            // get gray ROI and BGR pixel at corner of border rectangle in source image
            cv::Rect roi(pt2, sz_roi);
            cv::Mat img_roi = img_gray(roi);
            cv::Scalar bgr_pixel = img.at<cv::Vec3b>(pt1);

            // these should all match because they were captured with same settings
            cv::Mat img_match;
            std::vector<BGRLandmark::landmark_info_t> lminfo;
            bgrm.perform_match(img_roi, img_match, lminfo);

            // make a square image of fixed dim
            // convert to float in -128 to 127 range and run DCT on it (just like a JPEG block)
            cv::Mat img_dct, img_src, img_src_32f;
            cv::resize(img_roi, img_src, { kdct,kdct });
            img_src.convertTo(img_src_32f, CV_32F);
            img_src_32f -= 128.0;
            cv::dct(img_src_32f, img_dct, 0);

            // create a feature vector for this sample
            // by extracting components from the DCT
            std::vector<float> vfeature;
            for (int ii = kdctmincomp; ii <= kdctmaxcomp; ii++)
            {
                float val = img_dct.at<float>(vzigzagPts[ii]);
                vfeature.push_back(val);
            }

            // stick feature vector in appropriate structure
            if ((bgr_pixel != cv::Scalar{ 255, 255, 255 }) || (lminfo.size() == 0))
            {
                // non-white border (or no match from BGRLandmark) is junk sample
                vdat0.push_back(vfeature);
            }
            else if (lminfo[0].diff < 0)
            {
                // this is a "negative" sample
                vdatn.push_back(vfeature);
            }
            else
            {
                // this is a "positive" sample
                vdatp.push_back(vfeature);
            }
        }
    }

    // the samples have a crude ordering based on how they were collected
    // so shuffle in case we don't want to use all the samples
    // this insures a subset has similar variation
    std::random_shuffle(vdatp.begin(), vdatp.end());
    std::random_shuffle(vdatn.begin(), vdatn.end());
    std::random_shuffle(vdat0.begin(), vdat0.end());

    spew_float_vecs_to_csv(rstag, "_p", vdatp);
    spew_float_vecs_to_csv(rstag, "_n", vdatn);
    spew_float_vecs_to_csv(rstag, "_0", vdat0);

    // some experimental stuff with PCA...

    cv::Mat img_pca;
    read_csv_into_mat("train_9x9_p.csv", img_pca);

    cv::PCA mypca(img_pca, cv::noArray(), cv::PCA::DATA_AS_ROW, 0.98);
    cv::FileStorage cvfs;
    cvfs.open("fudge.yaml", cv::FileStorage::WRITE);
    if (cvfs.isOpened())
    {
        mypca.write(cvfs);
        cvfs.release();
    }

    cv::Mat wug = mypca.project(vdatp[1]);
    cv::Mat wog = mypca.backProject(wug);

    cv::Mat img_idct = cv::Mat::zeros({ kdct, kdct }, CV_32F);
    int mm = 0;
    for (int ii = kdctmincomp; ii <= kdctmaxcomp; ii++)
    {
        img_idct.at<float>(vzigzagPts[ii]) = wog.at<float>(0, mm);
        mm++;
    }
    cv::Mat bug;
    cv::idct(img_idct, bug, 0);
    cv::normalize(bug, bug, 0, 255, cv::NORM_MINMAX);
    cv::Mat zug;
    bug.convertTo(zug, CV_8U);
    cv::imwrite("zug.png", zug);

    //std::ofstream ofs;
    //ofs.open("fug.csv");
    //ofs << cv::format(img_pca, cv::Formatter::FMT_CSV) << std::endl;
    //ofs.close();
    
    return result;
}


void run_pca()
{
}

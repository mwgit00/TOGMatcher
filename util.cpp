#include "Windows.h"

#include "opencv2/highgui.hpp"

#include <list>
#include <iostream>
#include <sstream>

#include "util.h"


void get_dir_list(
    const std::string& rsdir,
    const std::string& rspattern,
    std::list<std::string>& listOfFiles)
{
    std::string s = rsdir + "\\" + rspattern;

    WIN32_FIND_DATA search_data;
    memset(&search_data, 0, sizeof(WIN32_FIND_DATA));

    HANDLE handle = FindFirstFile(s.data(), &search_data);

    while (handle != INVALID_HANDLE_VALUE)
    {
        std::string sfile(search_data.cFileName);
        listOfFiles.push_back(rsdir + "\\" + sfile);
        if (FindNextFile(handle, &search_data) == FALSE)
        {
            break;
        }
    }

    FindClose(handle);
}


bool make_video(
    const double fps,
    const std::string& rspath,
    const std::list<std::string>& rListOfPNG)
{
    bool result = false;
    
    // determine size of frames
    const std::string& rs = rListOfPNG.front();
    cv::Mat img = cv::imread(rs);
    cv::Size img_sz = img.size();

    std::string sname = rspath + "\\movie.wmv";

    // build movie from separate frames
    cv::VideoWriter vw = cv::VideoWriter(sname,
        CV_FOURCC('W', 'M', 'V', '2'),
        fps, img_sz);

    if (vw.isOpened())
    {
        for (const auto& r : rListOfPNG)
        {
            cv::Mat img = cv::imread(r);
            vw.write(img);
        }
        
        result = true;
    }

    return result;
}

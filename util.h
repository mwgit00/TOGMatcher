#ifndef UTIL_H_
#define UTIL_H_

#include <string>
#include <list>

typedef struct
{
    double mag_thr;
    std::string sname;
} T_file_info;

// Get list of all files in a directory that match a pattern
void get_dir_list(
    const std::string& rsdir,
    const std::string& rspattern,
    std::list<std::string>& listOfFiles);

// Use OpenCV routine to make video from a list of files
bool make_video(
    const double fps,
    const std::string& rspath,
    const std::list<std::string>& rListOfPNG);

#endif // UTIL_H_

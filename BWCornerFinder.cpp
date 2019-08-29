#include "opencv2/highgui.hpp"

#include "BWCornerFinder.h"

BWCornerFinder::BWCornerFinder()
{

}


BWCornerFinder::~BWCornerFinder()
{

}


void BWCornerFinder::init(const int k)
{
	int xk;
	int xkh;
	xk = (k < 3) ? 3 : k;
	xk = (k > 35) ? 35 : k;
	xkh = xk / 2;
	tmpl_p = cv::Mat::zeros({ xk, xk }, CV_8U);
	tmpl_n = cv::Mat::zeros({ xk, xk }, CV_8U);

	cv::rectangle(tmpl_p, { xkh + 1, 0, xkh, xkh }, 255, -1);
	cv::rectangle(tmpl_p, { 0, xkh + 1, xkh, xkh }, 255, -1);
	cv::line(tmpl_p, { xkh, 0 }, { xkh, xk - 1 }, 127);
	cv::line(tmpl_p, { 0, xkh }, { xk - 1, xkh }, 127);

	cv::rectangle(tmpl_n, { 0, 0, xkh, xkh }, 255, -1);
	cv::rectangle(tmpl_n, { xkh, xkh, xk - 1, xk - 1 }, 255, -1);
	cv::line(tmpl_n, { xkh, 0 }, { xkh, xk - 1 }, 127);
	cv::line(tmpl_n, { 0, xkh }, { xk - 1, xkh }, 127);

    tmpl_offset.width = xk / 2;
    tmpl_offset.height = xk / 2;

    cv::imshow("foop", tmpl_p);
    cv::imshow("foon", tmpl_p);
}


void BWCornerFinder::perform_match(
	const cv::Mat& rsrc,
	cv::Mat& rtmatch)
{
	cv::Mat tmatch_p;
	cv::Mat tmatch_n;

	// perform match with positive and negative templates
	matchTemplate(rsrc, tmpl_p, tmatch_p, cv::TemplateMatchModes::TM_SQDIFF_NORMED);
	matchTemplate(rsrc, tmpl_n, tmatch_n, cv::TemplateMatchModes::TM_SQDIFF_NORMED);

	// combine results by subtracting
    rtmatch = tmatch_n - tmatch_p;
}
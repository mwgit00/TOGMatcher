#ifndef BW_CORNER_FINDER_
#define BW_CORNER_FINDER_

#include "opencv2/imgproc.hpp"


// Selects size for corner template
// Options are odd values in range 3-35
#define BWC_DEFAULT_KSIZE	(3)


class BWCornerFinder
{
public:
	BWCornerFinder();
	virtual ~BWCornerFinder();

	void init(const int k);

	void perform_match(
		const cv::Mat& rsrc,
		cv::Mat& rtmatch);

    const cv::Size& get_template_offset(void) const { return tmpl_offset; }

private:

	cv::Mat tmpl_p;
	cv::Mat tmpl_n;

    // Offset for centering template location
    cv::Size tmpl_offset;
};

#endif // BW_CORNER_FINDER_
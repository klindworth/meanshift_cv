#include "ms_cv.h"

#include "msImageProcessor.h"
#include <opencv2/core/core.hpp>

/**
 * @param src Image to segment
 * @param labels_dst cv::Mat where the (int) labels will be written in
 * @return Number of different labels
 */

int mean_shift_segmentation(const cv::Mat& src, cv::Mat& labels_dst, int spatial_variance, float color_variance, int minsize)
{
	msImageProcessor proc;
	proc.DefineImage(src.data, (src.channels() == 3 ? COLOR : GRAYSCALE), src.rows, src.cols);
	proc.Segment(spatial_variance,color_variance, minsize, MED_SPEEDUP);//HIGH_SPEEDUP, MED_SPEEDUP, NO_SPEEDUP; high: set speedupThreshold, otherwise the algorithm uses it uninitialized!

	labels_dst = cv::Mat(src.size(), CV_32SC1);
	int regions_count = proc.GetRegionsModified(labels_dst.data);

	return regions_count;
} 

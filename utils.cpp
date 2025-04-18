#include "utils.h"
#include <opencv2/opencv.hpp>

void preprocess_face(cv::Mat& face) {
    // You can add face preprocessing steps here if needed (e.g., resizing, normalization)
    cv::resize(face, face, cv::Size(160, 160));  // Resize for FaceNet
}

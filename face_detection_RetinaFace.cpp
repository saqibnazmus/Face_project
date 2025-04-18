#include "face_detection.h"
#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <iostream>

FaceDetector::FaceDetector(const std::string& model_path) {
    // Load RetinaFace model
    try {
        torch::jit::script::Module module = torch::jit::load(model_path);
        model = std::make_shared<torch::jit::script::Module>(module);
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the RetinaFace model\n";
        exit(1);
    }
}

std::vector<cv::Rect> FaceDetector::detect_faces(const cv::Mat& frame) {
    std::vector<cv::Rect> faces;

    // Convert frame to tensor
    cv::Mat input_blob = frame;
    cv::cvtColor(input_blob, input_blob, cv::COLOR_BGR2RGB);
    input_blob.convertTo(input_blob, CV_32F, 1.0 / 255);
    torch::Tensor tensor_input = torch::from_blob(input_blob.data, {1, input_blob.rows, input_blob.cols, 3}, torch::kByte);
    tensor_input = tensor_input.permute({0, 3, 1, 2}).to(torch::kCUDA).float();

    // Run inference
    torch::Tensor output = model->forward({tensor_input}).toTensor();

    // Parse the output (This part depends on RetinaFace output format)
    // You'll need to extract bounding boxes, confidence scores, and landmarks
    // Here, assuming the output is directly bounding boxes (for simplicity)
    auto boxes = output.slice(2, 0, 4).cpu();
    for (int i = 0; i < boxes.size(1); ++i) {
        float x1 = boxes[0][i].item<float>();
        float y1 = boxes[1][i].item<float>();
        float x2 = boxes[2][i].item<float>();
        float y2 = boxes[3][i].item<float>();
        faces.push_back(cv::Rect(x1, y1, x2 - x1, y2 - y1));
    }

    return faces;
}

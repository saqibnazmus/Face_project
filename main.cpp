#include <opencv2/opencv.hpp>
#include "face_detection.h"
#include "face_recognition.h"
#include <iostream>

int main() {
    // Load models
    FaceDetector face_detector("models/retinaface_model.onnx");
    FaceRecognizer face_recognizer("models/facenet_model.onnx");

    cv::VideoCapture cap(0); // Open webcam
    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open webcam!" << std::endl;
        return -1;
    }

    cv::Mat frame;
    while (cap.read(frame)) {
        std::vector<cv::Rect> faces = face_detector.detect_faces(frame);

        for (auto& face : faces) {
            cv::Mat face_roi = frame(face);
            std::vector<float> embeddings = face_recognizer.recognize(face_roi);

            // Compare embeddings to identify if this is a known face
            // For simplicity, we'll just print the embeddings
            std::cout << "Face Embeddings: ";
            for (const auto& val : embeddings) {
                std::cout << val << " ";
            }
            std::cout << std::endl;

            // Draw the bounding box around the face
            cv::rectangle(frame, face, cv::Scalar(0, 255, 0), 2);
        }

        cv::imshow("Face Recognition", frame);
        if (cv::waitKey(1) == 27) break; // ESC to quit
    }

    return 0;
}

#include<algorithm>
#include<torch/torch.h>
#include<torch/script.h>
#include<opencv2/opencv.hpp>
#include<opencv2/core/utility.hpp>
#include<iostream>
#include<memory>
#include<string>
#include<fstream>
#include<vector>
#include<list>
#include<cmath>
#include<ctime>
#include<algorithm>
#include<numeric>

#include "face_detection.h"
#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <iostream>


#define SUCCESS 0
#define FAILURE 1


using namespace std;
using namespace cv;


int main(int argc, const char* argv[])
{    
    CascadeClassifier cascade;
    cascade.load("D:\\Software\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_default.xml");
    //std::cout << "load detection model successfully \n";
    std::string inputName = "sample image";
    //cv::String path("D:/Dataset/FRVT_frontal_face/*.jpg");

    //std:ifstream f("D:/C++/Input/input.txt");
    //std::string line;
    //std::vector<std::string> mylist;
    //while (std::getline(f, line))
    //{
        //mylist.push_back(line);
        //std::cout << mylist.back() << std::endl;
    //}
    cv::String path1("D:/Dataset/FRVT_frontal_face/*.png");
    std::cout << path1 << std::endl;
    vector<cv::String>fn;
    vector<cv::Mat>data;
    char* ptr;
    std::vector<std::string> mylist2;
    cv::glob(path1, fn, true);
    int nextnum = 00001;
    for (size_t k = 0; k < 1131; ++k)
    {
        cv::Mat image = cv::imread(fn[k], IMREAD_COLOR);
        std::cout << "fn[k]:" << fn[k] << endl;
        std::string str1 = fn[k].substr(29);
        str1.erase(14, 4);
        std::cout << str1 << endl;
        if (image.empty())
        {
            cout << "Could not read" << inputName << endl;
            return 1;
        }
        else
        {
            Mat frame1 = image.clone();
            //string filepath =  "D:/C++/Template/";
            //string data = to_string(nextnum);
            //std::ofstream filename;
            //filename.open(filepath + data + ".template");
            //filename.close();
            //ofstream templatefile;
            //ifstream file(filename, std::ios::binary);
            try {
                auto x = createTemplate(frame1);
                //std::cout << "vector:" << x.size() << endl;
                string data = to_string(nextnum);

                ofstream file;
                string filepath = "D:/C++/FRVT_Template/";
                file.open(filepath + str1 + ".template");
                file << x;
                file.close();







                mylist2.push_back(filepath + data + ".template\n");
                //std::cout << "mylist:" << mylist2.size() << endl;
                nextnum = nextnum++;
            }
            catch (const c10::Error& e)
            {
                std::cerr << "error loading the model\n" << e.msg();
                return -1;
            }//catch done

        }
    }
    std::ofstream z;
    z.open("D:/C++/FRVT_Input/log.txt");
    z << mylist2;
    z.close();



   string path2 = "";
   vector<String> fn1;
   cv::glob(path2, fn1, true); // recurse
   for (size_t x = 0; x < fn1.size(); x = x + 2)
   {
       std::ifstream o(fn1[x]);

       std::vector<float> t1;
       double element;
       while (o >> element)
       {
          t1.push_back(element);

       }
       std::cout << "t1: " << t1 << std::endl;
       o.close();

       std::ifstream p(fn1[x + 1]);
       std::vector<float> t2;
       double element1;
       while (p >> element1)
       {
          t2.push_back(element1);

       }
       std::cout << "t2: " << t2 << std::endl;
       p.close();
       float MSE = matchTemplate(t1, t2);
       std::vector<float>finalist;
       finalist.push_back(MSE);
       std::cout << "finalist:" << finalist << endl;

       std::ofstream z1;
       z1.open("D:/C++/FRVT_Input/FRVT_match.txt");
       z1 << finalist;

   }

}

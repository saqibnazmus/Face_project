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



#define SUCCESS 0
#define FAILURE 1


using namespace std;
using namespace cv;


int main(int argc, const char* argv[])
{
  string path2 = "D:/C++/FRVT_template/*.template";
    vector<String> fn1;
    cv::glob(path2, fn1, true);
    //std::cout << "length of path:" << fn1.size() << endl;
    for (size_t i = 0; i < fn1.size(); i++)
    {
       for (size_t j = 0; j < fn1.size(); j++)
        {
            string x = fn1[i];
            string y = fn1[j];
            std::vector<float>v1;
            std::vector<float>v2;
            v1 = float_extraction(x);
            v2 = float_extraction(y);
            //std::cout << "template1  :" << v1 << endl;
            //std::cout << "template2  :" << v2 << endl;
            //std::cout << "difference:" << v1-v2 << std::endl;
            std::string xc = x.substr(21);
            std::string xcut = xc.erase(14);
            //std::cout << "xcut:" << xcut << endl;
            std::string yc = y.substr(21);
            std::string ycut = yc.erase(14);
            //std::cout << "ycut:" << ycut << endl;
            std::string yfinal = xcut + "__" + ycut;
            std::cout << "final" << yfinal << endl;
             //float difference = v1 - v2;
                //double difference = matchTemplate(v1, v2);
                //std::vector<float>sum = 0;
            float sum;
            float differ = 0.0;
            //const int local_featureVectorSize = 512;
            std::vector<float>diff;
            for (size_t k = 0; k < v1.size(); k++)
            {
                sum = v1[k] - v2[k];
                differ += sum * sum;

            }

            //double sum_of_elements = std::accumulate(diff.begin(), diff.end(), 0);
            float av = v1.size();
            double result = (differ / av)*0.1;

            //cout.setf(ios::fixed, ios::floatfield);
            //cout.setf(ios::showpoint);
            std::cout << "difference:" << result << std::endl;
            std::string filepath = "D:/C++/FRVT_diff/";
            ofstream file;
            file.open(filepath + yfinal + ".txt");

            file << result;
            file.close();

}

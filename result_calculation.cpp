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







auto createTemplate(cv::Mat image) {
    CascadeClassifier cascade;
    cascade.load("D:\\Software\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_default.xml");
    Mat frame = image.clone();
    cv::Mat frame1 = detectAndCrop(frame, cascade);
    std::cout << "frame:" << frame1.size() << endl;
    at::Tensor tensor_cropped = ToTensor(frame1);
    tensor_cropped = transpose(tensor_cropped, { (2), (0), (1) }); 
    auto tensor_cropped1 = tensor_cropped.unsqueeze(0);
    tensor_cropped = tensor_cropped1.toType(c10::kFloat).div(255);
    auto inputs1 = std::vector<torch::jit::IValue>{ tensor_cropped };
    std::string r18_model_path = "D:\\C++\\demo1\\recognition.pt";
    torch::jit::script::Module r18;
    r18 = torch::jit::load(r18_model_path);
    at::Tensor cropped_tensor = r18.forward(inputs1).toTensor();
    cropped_tensor = cropped_tensor.contiguous();
    std::vector<float> v(cropped_tensor.data_ptr<float>(), cropped_tensor.data_ptr<float>() + cropped_tensor.numel());
    std::cout << "Vector:" << v.size() << endl;
    return v;
}



double matchTemplate(vector<float>& veriftemplate, vector<float>& enrolltemplate)
{
    float sum;
    float differ = 0.0;
    //const int local_featureVectorSize = 512;
    std::vector<float>diff;
   //double sum_of_elements = std::accumulate(diff.begin(), diff.end(), 0);
    float av = veriftemplate.size();
    float result = differ / av;

    return result;
}

  
auto float_extraction(const::string& str1)
{
    std::ifstream o(str1);
    std::vector<float>t1;
    float element;
    while (o >> element)
    {
        t1.push_back(element);
    }

    //std::cout << "t1:" << t1 << std::endl;
    o.close();
    return t1;
}



int main(int argc, const char* argv[])
{
    string path3 = "D:/C++/FRVT_diff/*.txt";
    vector<String> fn2;
    cv::glob(path3, fn2, true);
    int nextnum = 000001;
    int nextnumber = 00000001;
    double a = 0000001;
    double b = 0000001;
    double c = 0000001;
    double d = 0000001;
    for (size_t i = 0; i < 2; i++)
    {
        string m = fn2[i];
        std::ifstream infile(m);
        int spec_no = 10;
        int ion_no = 10;
        






       // std::cout << m << endl;
        std::string mc = m.substr(17);
        std::string mcut = mc.erase(4);
        //std::cout << "mcut:" << mcut << endl;
        std::string mcc = m.substr(33);
        std::string mccut = mcc.erase(4);
        //std::cout << "mccut:" << mccut << endl;
        std::ifstream r(m);
        std::vector<float> z1;
        float elem;
        while (r >> elem)
        {
            z1.push_back(elem);
        }
        //std::cout << "z1:" << z1 << endl;        //z1 = float_extraction(m);
        std::vector<float>same_class;
        std::vector<float>diff_class;
        
        if (mcut == mccut)
        {
            same_class.push_back(elem);
            nextnum = nextnum++;
            //std::cout << "These are same class" << endl;
        }
        else
        {
            diff_class.push_back(elem);
            nextnumber = nextnumber++;
            //std::cout << "These are different class" << endl;
        }

        float threshold = 0.00001;
        
        ///for same class///
        //false negative//
        //true positive//
        
        std::vector<float>fn_list;
        std::vector<float>tp_list;
        float result;
        float result2;
        
        for (size_t j = 0; j < same_class.size(); j++)
        {
           result = same_class[j];
           if (result > threshold)
           {
               fn_list.push_back(result);
               a = a++;
           }
        }
        for (size_t xx = 0; xx < same_class.size(); xx++)
        {
            result2 = same_class[xx];
            if (result2 <= threshold)
            {
                tp_list.push_back(result2);
                b = b++;
            }
        }

        ///for different class//
        //false positive//
        //true negative//

        std::vector<float>fp_list;
        std::vector<float>tn_list;
        float result1;
        float result3;
        for (size_t k = 0; k < diff_class.size(); k++)
        {
            result1 = diff_class[k];
            if (result1 < threshold)
            {
                fp_list.push_back(result1);
                c = c++;
            }
        }
        for (size_t xj = 0; xj < diff_class.size(); xj++)
        {
            result3 = diff_class[xj];
            if (result3 >= threshold)
            {
                tn_list.push_back(result2);
                d = d++;
            }
        }
    }

    
    std::cout << "length of same class:" << nextnum << endl;
    std::cout << "length of different class:" << nextnumber << endl;
    std::cout << "length of false negative list:" << a << endl;
    std::cout << "length of false positive list:" << c << endl;
    std::cout << "length of true positive list:" << b << endl;
    std::cout << "length of true negative list:" << d << endl;

    double FMR = double(c)/double(c+d);
    double FNMR = double(a) / double(a + b);
    double FRVT_Accuracy = double(b + d) / double(a + b + c + d);

    std::cout << "FMR Result:" << FMR << endl;
    std::cout << "FNMR Result:" << FNMR << endl;
    std::cout << "FRVT Accuracy:" << FRVT_Accuracy << endl;
    //std::cout << "Your processing done\n";
    //std::system("pause");
    return 0;

        
}
    

}


#include <iostream>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

int hmin_slider;
int key;

int hmin, hmax, satmin, satmax, valmin, valmax;

static void huemin_callback(int, void*){}
static void satmin_callback(int, void*){}
static void valmin_callback(int, void*){}
static void huemax_callback(int, void*){}
static void satmax_callback(int, void*){}
static void valmax_callback(int, void*){}

int main(void)
{
    // trackbars e variables para almacenar o valor dos trackbars
    cv::namedWindow("Deslizadores", cv::WINDOW_AUTOSIZE);

    cv::createTrackbar("Hue min", "Deslizadores", &hmin, 255, huemin_callback);
    cv::createTrackbar("Sat min", "Deslizadores", &satmin, 255, satmin_callback);
    cv::createTrackbar("Val min", "Deslizadores", &valmin, 255, valmin_callback);
    cv::createTrackbar("Hue max", "Deslizadores", &hmax, 255, huemax_callback);
    cv::createTrackbar("Sat max", "Deslizadores", &satmax, 255, satmax_callback);
    cv::createTrackbar("Val max", "Deslizadores", &valmax, 255, valmax_callback);

    // creamos mascara
    cv::Mat mask, hsv, image;
    cv::namedWindow("Mascara");
    
    // lemos a imaxe
    image = cv::imread("../barco.jpg");
    
    // convertimos color
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);

    while (1)
    {
        // creamos unha mascara
        cv::inRange(hsv, cv::Scalar(hmin, satmin, valmin), cv::Scalar(hmax, satmax, valmax), mask);
        cv::imshow("Mascara", mask);

        // facemos un bitwise entre a máscara e a imaxe
        cv::Mat bitwise;
        cv::bitwise_and(image, image, bitwise, mask);
        cv::namedWindow("bitwise and");
        cv::imshow("bitwise and", bitwise);

        // control
        char key = cv::waitKey(30);
        if (key == 's')
        {
            break;
        }
        
    }
    
    return 0;
}
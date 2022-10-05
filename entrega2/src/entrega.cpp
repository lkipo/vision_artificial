#include <iostream>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>

int hmin_slider;
int key;

static void callback(int, void*){}

int main(void)
{
    // lemos a imaxe
    cv::Mat image;
    image = cv::imread("../barco.jpg");

    // mostramos a imaxe
    cv::namedWindow("Display image", cv::WINDOW_AUTOSIZE);
    cv::imshow("Display image", image);
    
    // creamos ventana para os sliders
    cv::namedWindow("Deslizadores", cv::WINDOW_AUTOSIZE);
    hmin_slider = 0;
    cv::createTrackbar("Hue min", "Deslizadores", &hmin_slider, 255, callback);

    // creamos mascara
    cv::Mat mask;
    cv::namedWindow("Mascara");

    while (key!='s')
    {
        cv::inRange(image, (0, 0, 0), (200, 200, 200), mask);
        // std::cout << hmin_slider << '\n'; 
        cv::waitKeyEx(20); // Cambiar esto

        // mostramos a mascara
        cv::imshow("Mascara", mask);
    }
    return 0;
}
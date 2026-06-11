/*
 * This file is part of the JuMarker library
 * Copyright (c) 2022 David Jurado Rodríguez, Rafael Muñoz Salinas,
 *                    Sergio Garrido Jurado, Rafael Medina Carnicer.
 *
 * JuMarker is published and distributed under the CC BY-NC-SA license.
 * JuMarker is distributed in the hope that it will be useful for
 * non-commercial academic research, but WITHOUT ANY WARRANTY.
 *
 * You should have received a copy of the CC BY-NC-SA license along with
 * this program; if not, write to rmsalinas@uco.es.
 */

#include "svgData.h"

using namespace std;

class CmdLineParser
{
    int argc;
    char **argv;

  public:
    CmdLineParser(int _argc, char **_argv) : argc(_argc), argv(_argv)
    {
    }
    bool operator[](string param)
    {
        int idx = -1;
        for (int i = 0; i < argc && idx == -1; i++)
            if (string(argv[i]) == param)
                idx = i;
        return (idx != -1);
    }
    string operator()(string param, string defvalue = "-1")
    {
        int idx = -1;
        for (int i = 0; i < argc && idx == -1; i++)
            if (string(argv[i]) == param)
                idx = i;
        if (idx == -1)
            return defvalue;
        else
            return (argv[idx + 1]);
    }
};
int markerNumber;
int main(int argc, char **argv)
{
    try
    {
        CmdLineParser cml(argc, argv);
        if (argc < 3 || cml["-h"])
        {
            cerr << "Invalid number of arguments" << endl;
            cerr << "Usage: [path svg file] [path directory] [markers number]" << endl;
            return false;
        }

        // Instanciate SVGData object
        SVGData fileData;
        // read SVG
        SVGParse::read(argv[1], fileData);

        int highMarkerNumber = cv::pow(2, fileData.elements.size() - 16);
        markerNumber = std::stoi(argv[3]);
        if (markerNumber > highMarkerNumber)
        {
            markerNumber = highMarkerNumber;
            cout << "It has been possible to create only " << highMarkerNumber << " markers " << endl;
        }

        // Bits user according to markers number input
        int bitsElements = log(markerNumber) / log(2);

        // Create multiple svg files with unique id
        SVGParse::save(argv[1], argv[2], bitsElements, fileData);

        cout << "Total bits: " << fileData.elements.size() << " / Bits used for code data: " << bitsElements
             << " / Bits used for CRC: " << 16 << endl;
        if (fileData.elements.size() - (bitsElements + 16) != 0)
            cout << "Bits not assigned: " << fileData.elements.size() - (bitsElements + 16) << endl;

        // Instanciate window
        cv::Mat sgv_img = cv::Mat::zeros(700, 700, CV_8UC3);
        sgv_img.setTo(cv::Scalar(255, 255, 255));

        // Draw marker border
        fileData.drawBoder(sgv_img);

        // Draw marker elements
        fileData.drawElements(sgv_img);

        cv::imshow("SVG Test", sgv_img);
        cv::waitKey(0);
    }
    catch (std::exception &ex)
    {
        cout << "Exception :" << ex.what() << endl;
    }
}

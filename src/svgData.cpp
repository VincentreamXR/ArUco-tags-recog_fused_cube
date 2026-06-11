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
#define CRC16 0x8005
string layerContour = "contour";
string layerCode = "code";
const int contratColor = 60;
const string folderPngMarker = "/pngImages";

std::vector<std::string> SplitWithCharacters(const std::string &str, int splitLength)
{
    int NumSubstrings = str.length() / splitLength;
    std::vector<std::string> ret;

    for (int i = 0; i < NumSubstrings; i++)
    {
        ret.push_back(str.substr(i * splitLength, splitLength));
    }

    // If there are leftover characters, create a shorter item at the end.
    if (str.length() % splitLength != 0)
    {
        ret.push_back(str.substr(splitLength * NumSubstrings));
    }
    return ret;
}

cv::Point2f SVGData::getCenter()
{
    if (border.size() > 2)
    {
        float doubleArea = 0;
        cv::Point2f p(0, 0);
        cv::Point2f p0 = border.back();
        for (const cv::Point2f &p1 : border)
        {                                        // C++11
            float a = p0.x * p1.y - p0.y * p1.x; // cross product, (signed) double area of triangle of vertices (origin,p0,p1)
            p += (p0 + p1) * a;
            doubleArea += a;
            p0 = p1;
        }
        if (doubleArea != 0)
            return p * (1 / (3 * doubleArea)); // Operator / does not exist for cv::Point
    }
    return cv::Point2f();
}

vector<string> split1(const string &str, const string &delim)
{
    vector<string> tokens;
    size_t prev = 0, pos = 0;
    do
    {
        pos = str.find(delim, prev);
        if (pos == string::npos)
            pos = str.length();
        string token = str.substr(prev, pos - prev);
        if (!token.empty())
            tokens.push_back(token);
        prev = pos + delim.length();
    } while (pos < str.length() && prev < str.length());
    return tokens;
}

std::vector<std::string> split(const std::string &stringToSplit, const std::string &regexString)
{
    std::regex regexToSplitWith{regexString};
    // create a ForwardIterator of submatches for the regex regexToSplitWith
    std::sregex_token_iterator wordsIterator{stringToSplit.begin() + 2, stringToSplit.end() - 1, regexToSplitWith, -1};

    // return rvalue vector to utilize vector move semantics to avoid copy
    return {wordsIterator, std::sregex_token_iterator{}};
}

void SVGData::drawBoder(cv::Mat &Image)
{
    for (int i = 0; i < border.size(); i++)
    {
        cv::Point2f p1;
        cv::Point2f p2;
        if (i == border.size() - 1)
        {
            p1 = border.at(i);
            p2 = border.at(0);
        }
        else
        {
            p1 = border.at(i);
            p2 = border.at(i + 1);
        }
        cv::line(Image, p1, p2, cv::Scalar(0, 0, 0), 2);
        cv::circle(Image, p1, 2, cv::Scalar(255, 0, 0), 5, cv::LINE_8, 0);
    }
}

void SVGData::drawElements(cv::Mat &Image)
{
    for (int i = 0; i < this->elements.size(); i++)
    {
        cv::Point2f p1 = this->elements.at(i).center;
        cv::circle(Image, p1, 2, cv::Scalar(0, 0, 255), -1, cv::LINE_8, 0);
    }
}

inline cv::Point2f computeCentroid(std::vector<cv::Point2f> &in)
{
    if (in.size() > 2)
    {
        float doubleArea = 0;
        cv::Point2f p(0, 0);
        cv::Point2f p0 = in.back();
        for (const cv::Point2f &p1 : in)
        {                                        // C++11
            float a = p0.x * p1.y - p0.y * p1.x; // cross product, (signed) double area of triangle of vertices (origin,p0,p1)
            p += (p0 + p1) * a;
            doubleArea += a;
            p0 = p1;
        }
        if (doubleArea != 0)
            return p * (1 / (3 * doubleArea)); // Operator / does not exist for cv::Point
    }
    return cv::Point2f();
}

bool clockwise(vector<cv::Point2f> vertices)
{
    double sum = 0.0;
    for (int i = 0; i < vertices.size(); i++)
    {
        cv::Point2f v1 = vertices[i];
        cv::Point2f v2 = vertices[(i + 1) % vertices.size()];
        sum += (v2.x - v1.x) * (v2.y + v1.y);
    }
    return sum < 0.0;
}

cv::Scalar hex2rgb(string hex)
{
    cv::Scalar color;

    if (hex.at(0) == '#')
    {
        hex.erase(0, 1);
    }

    while (hex.length() != 6)
    {
        hex += "0";
    }

    std::vector<string> colori = SplitWithCharacters(hex, 2);

    color[0] = stoi(colori[0], nullptr, 16);
    color[1] = stoi(colori[1], nullptr, 16);
    color[2] = stoi(colori[2], nullptr, 16);

    return color;
}

bool chechSymmetry(const std::vector<cv::Point2f> &markerBorder)
{
    if (markerBorder.size() <= 4)
        return true;

    std::vector<cv::Point2f> rotate;

    for (size_t i = 1; i < markerBorder.size(); i++)
    {
        rotate = markerBorder;

        std::rotate(rotate.begin(), rotate.begin() + i, rotate.end());

        cv::Mat R = cv::estimateAffine2D(rotate, markerBorder);

        // extend rigid transformation to use perspectiveTransform:
        cv::Mat H = cv::Mat(3, 3, R.type());
        H.at<double>(0, 0) = R.at<double>(0, 0);
        H.at<double>(0, 1) = R.at<double>(0, 1);
        H.at<double>(0, 2) = R.at<double>(0, 2);

        H.at<double>(1, 0) = R.at<double>(1, 0);
        H.at<double>(1, 1) = R.at<double>(1, 1);
        H.at<double>(1, 2) = R.at<double>(1, 2);

        H.at<double>(2, 0) = 0.0;
        H.at<double>(2, 1) = 0.0;
        H.at<double>(2, 2) = 1.0;

        bool rotationSymmetry = true;
        for (size_t j = 0; j < markerBorder.size(); j++)
        {
            cv::Point2f auxVertex = H * rotate[j];

            if ((int)norm(auxVertex - markerBorder[j]) != 0)
            {
                rotationSymmetry = false;
                break;
            }
        }
        if (rotationSymmetry)
            return true;
    }
    return false;
}

void SVGParse::read(string filename, SVGData &SVGDataFile)
{
    boost::property_tree::ptree pt;
    boost::property_tree::xml_parser::read_xml(filename, pt, boost::property_tree::xml_parser::trim_whitespace);
    boost::property_tree::ptree root = pt.get_child("svg");

    for (auto const &child : root)
    {
        if (child.first == "g")
        {
            // cout << child.second.get<string>("<xmlattr>.id") << endl;
            boost::property_tree::ptree subtree = child.second;
            if (child.second.get<string>("<xmlattr>.id") == layerContour)
            {
                for (auto const &st : subtree)
                {
                    if (st.first == "path")
                    {
                        string row = st.second.get<string>("<xmlattr>.d");
                        vector<std::string> line = split(row, "[\\s,]+");
                        for (int i = 0; i < line.size(); i += 2)
                        {
                            if (line[i] == "H")
                            {
                                /* P = (x , y back)*/
                                SVGDataFile.border.push_back(
                                    cv::Point2f(atof(line[i + 1].c_str()), SVGDataFile.border[SVGDataFile.border.size() - 1].y));
                            }

                            else if (line[i] == "L")
                            {
                                /* P = (x, y)*/
                                SVGDataFile.border.push_back(cv::Point2f(atof(line[i + 1].c_str()), atof(line[i + 2].c_str())));
                                i++;
                            }
                            else if (line[i] == "V")
                            {
                                /* P = (x back , y)*/
                                SVGDataFile.border.push_back(
                                    cv::Point2f(SVGDataFile.border[SVGDataFile.border.size() - 1].x, atof(line[i + 1].c_str())));
                            }
                            else
                                SVGDataFile.border.push_back(cv::Point2f(atof(line[i].c_str()), atof(line[i + 1].c_str())));
                        }
                    }
                    else if (st.first == "rect")
                    {
                        string x = st.second.get<string>("<xmlattr>.x");
                        string y = st.second.get<string>("<xmlattr>.y");
                        string width = st.second.get<string>("<xmlattr>.width");
                        string height = st.second.get<string>("<xmlattr>.height");

                        SVGDataFile.border.push_back(cv::Point2f(atof(x.c_str()), atof(y.c_str())));
                        SVGDataFile.border.push_back(cv::Point2f(atof(x.c_str()) + atof(width.c_str()), atof(y.c_str())));
                        SVGDataFile.border.push_back(
                            cv::Point2f(atof(x.c_str()) + atof(width.c_str()), atof(y.c_str()) + atof(height.c_str())));
                        SVGDataFile.border.push_back(cv::Point2f(atof(x.c_str()), atof(y.c_str()) + atof(height.c_str())));
                    }
                }
            }

            else if (child.second.get<string>("<xmlattr>.id") == layerCode)
            {
                vector<string> differentColor;
                for (auto const &st : subtree)
                {
                    if (st.first == "path")
                    {

                        /// READ POINT
                        string row = st.second.get<string>("<xmlattr>.d");
                        vector<std::string> line = split(row, "[\\s,]+");
                        vector<cv::Point2f> points;
                        cv::Point2f des(0, 0);
                        for (int i = 0; i < line.size(); i += 2)
                        {
                            // Relative coordinates
                            if (row[0] == 'm')
                            {
                                if (line[i] == "v" || line[i] == "V" || line[i] == "H" || line[i] == "h" || line[i] == "l" ||
                                    line[i] == "v")
                                {
                                    cout << "ERROR: SAVE FILE IN ABSOLUTE COORDINATE" << endl;
                                }

                                cv::Point2f newPoint = cv::Point2f(atof(line[i].c_str()), atof(line[i + 1].c_str()));
                                points.push_back(newPoint + des);
                                des += cv::Point2f(atof(line[i].c_str()), atof(line[i + 1].c_str()));
                            }
                            // Absolute coordinates
                            else if (row[0] == 'M')
                            {

                                if (line[i] == "v")
                                {
                                    /* P = (x back, y + dy)*/
                                    points.push_back(cv::Point2f(points[points.size() - 1].x,
                                                                 points[points.size() - 1].y + atof(line[i + 1].c_str())));
                                }

                                else if (line[i] == "V")
                                {
                                    /* P = (x back , y)*/
                                    points.push_back(cv::Point2f(points[points.size() - 1].x, atof(line[i + 1].c_str())));
                                }

                                else if (line[i] == "h")
                                {
                                    /* P = (x + dx, y back)*/
                                    points.push_back(cv::Point2f(points[points.size() - 1].x - atof(line[i + 1].c_str()),
                                                                 points[points.size() - 1].y));
                                }

                                else if (line[i] == "H")
                                {
                                    /* P = (x , y back)*/
                                    points.push_back(cv::Point2f(atof(line[i + 1].c_str()), points[points.size() - 1].y));
                                }

                                else if (line[i] == "L")
                                {
                                    /* P = (x, y)*/
                                    points.push_back(cv::Point2f(atof(line[i + 1].c_str()), atof(line[i + 2].c_str())));
                                    i++;
                                }

                                else
                                {
                                    points.push_back(cv::Point2f(atof(line[i].c_str()), atof(line[i + 1].c_str())));
                                }
                            }
                        }

                        /// READ COLOR
                        string lineStyle = st.second.get<string>("<xmlattr>.style");
                        std::size_t found = lineStyle.find("fill:#");
                        string colorH;
                        for (unsigned i = found + 6; i < found + 12; i++)
                        {
                            colorH.push_back(lineStyle.at(i));
                        }

                        // Detect differents colors in svg file
                        if (std::find(differentColor.begin(), differentColor.end(), colorH) == differentColor.end())
                        {
                            differentColor.push_back(colorH);
                        }

                        // Add individual code
                        SVGDataFile.elements.push_back(CodeElement(colorH, computeCentroid(points)));
                    }

                    else if (st.first == "rect")
                    {

                        string x = st.second.get<string>("<xmlattr>.x");
                        string y = st.second.get<string>("<xmlattr>.y");
                        string width = st.second.get<string>("<xmlattr>.width");
                        string height = st.second.get<string>("<xmlattr>.height");

                        int cx = atof(x.c_str()) + atof(width.c_str()) / 2;
                        int cy = atof(y.c_str()) + atof(height.c_str()) / 2;

                        // Read color
                        string lineStyle = st.second.get<string>("<xmlattr>.style");
                        std::size_t found = lineStyle.find("fill:#");
                        string colorH;

                        for (unsigned i = found + 6; i < found + 12; i++)
                        {
                            colorH.push_back(lineStyle.at(i));
                        }

                        // Detect differents colors in svg file
                        if (std::find(differentColor.begin(), differentColor.end(), colorH) == differentColor.end())
                        {
                            differentColor.push_back(colorH);
                        }

                        SVGDataFile.elements.push_back(CodeElement(colorH, cv::Point2f(cx, cy)));
                    }

                    else if (st.first == "ellipse" || st.first == "circle")
                    {
                        int cx = atof(st.second.get<string>("<xmlattr>.cx").c_str());
                        int cy = atof(st.second.get<string>("<xmlattr>.cy").c_str());

                        string lineStyle = st.second.get<string>("<xmlattr>.style");
                        std::size_t found = lineStyle.find("fill:#");
                        string colorH;
                        for (unsigned i = found + 6; i < found + 12; i++)
                        {
                            colorH.push_back(lineStyle.at(i));
                        }

                        // Detect differents colors in svg file
                        if (std::find(differentColor.begin(), differentColor.end(), colorH) == differentColor.end())
                        {
                            differentColor.push_back(colorH);
                        }

                        SVGDataFile.elements.push_back(CodeElement(colorH, cv::Point2f(cx, cy)));
                    }
                }

                SVGDataFile.colorBit1 = differentColor[1]; // White color
                SVGDataFile.colorBit0 = differentColor[0]; // Black color

                cv::Scalar color1 = hex2rgb(differentColor[1]);
                cv::Scalar color0 = hex2rgb(differentColor[0]);

                float diffGrayColor = abs((color1[0] + color1[1] + color1[2]) / 3 - (color0[0] + color0[1] + color0[2]) / 3);

                cout << "Diff gray color: " << diffGrayColor << endl;

                if (diffGrayColor < contratColor)
                    SVGDataFile.HSV = true;

                else
                    SVGDataFile.HSV = false;
            }
        }
    }

    // Check if marker is a convex polygon
    if (isContourConvex(SVGDataFile.border))
    {
        cout << "convexPolygon: Yes" << endl;
        SVGDataFile.convexPolygon = true;
    }
    else
    {
        cout << "convexPolygon: No" << endl;
        SVGDataFile.convexPolygon = false;
    }

    // Check if marker border are in clockwise
    if (!clockwise(SVGDataFile.border))
        std::reverse(SVGDataFile.border.begin(), SVGDataFile.border.end());

    // Check if marker is symetric
    if (chechSymmetry(SVGDataFile.border))
    {
        cout << "Symmetry: Yes" << endl;
        SVGDataFile.symmetry = true;
    }
    else
    {
        cout << "Symmetry: No" << endl;
        SVGDataFile.symmetry = false;
    }

    // SVGDataFile.numberBitsData = SVGDataFile.elements.size() -16;
}

inline uint16_t crc16(uint8_t const *data, size_t size)
{
    uint16_t crc = 0;
    while (size--)
    {
        crc ^= *data++;
        for (unsigned k = 0; k < 8; k++)
            crc = crc & 1 ? (crc >> 1) ^ 0xa001 : crc >> 1;
    }
    return crc;
}

inline string uint16ToBinary(uint16_t n)
{
    char buffer[17]; /* 16 bits, plus room for a \0 */
    buffer[16] = '\0';
    for (int i = 15; i >= 0; --i)
    {                              /* convert bits from the end */
        buffer[i] = '0' + (n & 1); /* '0' + bit => '0' or '1' */
        n >>= 1;                   /* make the next bit the 'low bit' */
    }
    return buffer;
}

void SVGParse::save(string file, string dir, int BitsMarkersUsed, SVGData &markerDesign)
{

    std::size_t found = file.find_last_of("/\\");
    vector<std::string> lineDel = split1(file.substr(found + 1), ".");
    string nameMarker = lineDel[0];

    for (int i = 1; i < pow(2, BitsMarkersUsed) + 1; i++)
    {

        int id = i;

        // file name
        string newFile = dir + nameMarker + "_" + to_string(id) + ".svg";

        // transform id to binary code
        boost::dynamic_bitset<> idCode = boost::dynamic_bitset<>(BitsMarkersUsed, id);

        // String code transform
        string buffer;
        to_string(idCode, buffer);

        // uint8_t code transform
        const uint8_t *p = reinterpret_cast<const uint8_t *>(buffer.c_str());

        // Final string CRC16 bits
        string CRC = uint16ToBinary(crc16(p, BitsMarkersUsed));

        // create an empty tree
        boost::property_tree::ptree tree;

        // Start reading svg file
        read_xml(file, tree, boost::property_tree::xml_parser::trim_whitespace);

        string totalColorsValue = buffer + CRC;

        system(("mkdir -p " + dir + folderPngMarker).c_str());

        for (auto &child : tree.get_child("svg"))
        {
            if (child.first == "g")
            {
                // Only change code calues
                if (child.second.get<string>("<xmlattr>.id") == layerCode)
                {
                    int posCodeData = 0; // to control return position 0 when it's necessary
                    int bitsChange = 0;  // to known how many markers have been changed
                    int bitsCRC = 0;

                    for (auto &st : child.second)
                    {
                        if (bitsChange < totalColorsValue.size())
                        {
                            if (st.first == "path" || st.first == "rect" || st.first == "ellipse" || st.first == "circle")
                            {

                                // Read color
                                string lineStyle = st.second.get<string>("<xmlattr>.style");
                                found = lineStyle.find("fill:#");
                                string colorH;

                                for (unsigned i = found + 6; i < found + 12; i++)
                                {
                                    colorH.push_back(lineStyle.at(i));
                                }

                                if (posCodeData < BitsMarkersUsed)
                                {
                                    if (buffer[posCodeData] == '1')
                                    {
                                        boost::replace_all(lineStyle, "fill:#" + colorH, "fill:#" + markerDesign.colorBit1);
                                    }
                                    else if (buffer[posCodeData] == '0')
                                    {
                                        boost::replace_all(lineStyle, "fill:#" + colorH, "fill:#" + markerDesign.colorBit0);
                                    }
                                    posCodeData++;
                                }

                                else
                                {
                                    if (CRC[bitsCRC] == '1')
                                    {
                                        boost::replace_all(lineStyle, "fill:#" + colorH, "fill:#" + markerDesign.colorBit1);
                                    }
                                    else if (CRC[bitsCRC] == '0')
                                    {
                                        boost::replace_all(lineStyle, "fill:#" + colorH, "fill:#" + markerDesign.colorBit0);
                                    }
                                    bitsCRC++;
                                }
                                // Change style value
                                st.second.put("<xmlattr>.style", lineStyle);
                                bitsChange++;
                            }
                        }
                    }
                }
            }
        }
        // Write new file
        write_xml(newFile, tree, std::locale(), boost::property_tree::xml_writer_make_settings<std::string>(' ', 4));
        string str = "flatpak run org.inkscape.Inkscape " + newFile + " -o " + dir + folderPngMarker + "/" + nameMarker + "_" +
                     to_string(id) + ".png";
        system(str.c_str());

        cout << "New svg file with name: " << newFile << " id: " << i << " code: " << idCode << " CRC: " << CRC << endl;
    }
}

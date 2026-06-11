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

#ifndef Hash_H_
#define Hash_H_
#include <cstdint>
#include <iostream>
#include <opencv2/core/core.hpp>
namespace ucoslam
{
/**
 * @brief The Hash struct creates a hash by adding elements. It is used to check the integrity of data stored in files in debug
 * mode
 */
struct Hash
{
    uint64_t seed = 0;

    template <typename T> void add(const T &val)
    {
        char *p = (char *)&val;
        for (uint32_t b = 0; b < sizeof(T); b++)
            seed ^= p[b] + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    template <typename T> void add(const T &begin, const T &end)
    {
        for (auto it = begin; it != end; it++)
            add(*it);
    }

    void add(bool val)
    {
        seed ^= int(val) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    void add(int val)
    {
        seed ^= val + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    void add(uint64_t val)
    {
        seed ^= val + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    void add(int64_t val)
    {
        seed ^= val + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    void add(float val)
    {
        int *p = (int *)&val;
        seed ^= *p + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    void add(double val)
    {
        uint64_t *p = (uint64_t *)&val;
        seed ^= *p + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }

    void add(const cv::Mat &m)
    {
        for (int r = 0; r < m.rows; r++)
        {
            const char *ip = m.ptr<char>(r);
            int nem = m.elemSize() * m.cols;
            for (int i = 0; i < nem; i++)
                seed ^= ip[i] + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
    }
    void add(const cv::Point2f &p)
    {
        add(p.x);
        add(p.y);
    }
    void add(const cv::Point3f &p)
    {
        add(p.x);
        add(p.y);
        add(p.z);
    }

    void operator+=(bool v)
    {
        add(v);
    }
    void operator+=(int v)
    {
        add(v);
    }
    void operator+=(char v)
    {
        add(v);
    }
    void operator+=(float v)
    {
        add(v);
    }
    void operator+=(double v)
    {
        add(v);
    }
    void operator+=(uint32_t v)
    {
        add(v);
    }
    void operator+=(uint64_t v)
    {
        add(v);
    }
    void operator+=(int64_t v)
    {
        add(v);
    }
    void operator+=(const cv::Mat &v)
    {
        add(v);
    }
    void operator+=(const cv::Point2f &v)
    {
        add(v);
    }
    void operator+=(const cv::Point3f &v)
    {
        add(v);
    }
    void operator+=(const std::string &str)
    {
        for (const auto &c : str)
            add(c);
    }

    operator uint64_t() const
    {
        return seed;
    }

    std::string tostring()
    {
        return tostring(seed);
    }

    static std::string tostring(uint64_t v)
    {
        std::string sret;
        std::string alpha = "qwertyuiopasdfghjklzxcvbnm1234567890QWERTYUIOPASDFGHJKLZXCVBNM";
        unsigned char *s = (uchar *)&v;
        int n = sizeof(seed) / sizeof(uchar);
        for (int i = 0; i < n; i++)
        {
            sret.push_back(alpha[s[i] % alpha.size()]);
        }
        return sret;
    }
    void reset()
    {
        seed = 0;
    }
};

} // namespace ucoslam

#endif

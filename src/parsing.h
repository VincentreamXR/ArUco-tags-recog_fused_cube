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

#ifndef btwutils_BasicPARSING_H_IU139Ik13oL
#define btwutils_BasicPARSING_H_IU139Ik13oL

#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>
namespace btwutils
{

//#define ParamSetSepChatacters {' ','\t','=',';','\n'}

class ParamSet
{
    std::map<std::string, std::string> map;

  public:
    virtual ~ParamSet()
    {
    }

    ParamSet()
    {
    }
    ParamSet(const std::string &str)
    {
        *this = fromString(str);
    }
    void clear()
    {
        map.clear();
    }

    inline bool is(const std::string &key) const
    {
        return map.count(key) != 0;
    }

    inline std::map<std::string, std::string>::iterator begin()
    {
        return map.begin();
    }
    inline std::map<std::string, std::string>::iterator end()
    {
        return map.end();
    }
    inline std::map<std::string, std::string>::const_iterator begin() const
    {
        return map.begin();
    }
    inline std::map<std::string, std::string>::const_iterator end() const
    {
        return map.end();
    }

    inline void insert(std::pair<std::string, std::string> k)
    {
        map.insert(k);
    }

    inline std::string &operator[](const std::string &str)
    {
        return map[str];
    }

    inline std::string &at(const std::string &str)
    {
        return map.at(str);
    }

    inline void erase(const std::string &key)
    {
        map.erase(key);
    }

    std::string asString(const std::string &key, const std::string &defval = "") const
    {
        auto elem = map.find(key);
        if (elem == map.end())
            return defval;
        return elem->second;
    }

    int asInt(const std::string &key, int defVal = std::numeric_limits<int>::min()) const
    {
        auto elem = map.find(key);
        if (elem == map.end())
            return defVal;
        try
        {
            defVal = stoi(elem->second);
        }
        catch (const std::exception &ex)
        {
        }
        return defVal;
    }
    uint64_t asULLong(const std::string &key, uint64_t defVal = std::numeric_limits<uint64_t>::min()) const
    {
        auto elem = map.find(key);
        if (elem == map.end())
            return defVal;
        try
        {
            defVal = stoull(elem->second);
        }
        catch (const std::exception &ex)
        {
        }
        return defVal;
    }

    double asDouble(const std::string &key, double defVal = std::numeric_limits<double>::quiet_NaN()) const
    {
        auto elem = map.find(key);
        if (elem == map.end())
            return defVal;
        try
        {
            defVal = stod(elem->second);
        }
        catch (const std::exception &ex)
        {
        }

        return defVal;
    }

    std::vector<std::string> asStringList(const std::string &key, char sep = ',') const
    {
        auto elem = map.find(key);
        if (elem == map.end())
            return {};
        // use the , separator
        return parseAsVector(elem->second, {sep}, false);
    }
    template <typename T> static std::string toStringList(const std::vector<T> &list)
    {
        std::stringstream res;
        for (auto e : list)
            res << e << ",";
        return res.str();
    }

    template <typename T>
    std::vector<T> asList(const std::string &key, const std::vector<T> &def = {}, size_t minElms = 0, char sep = ',') const
    {
        auto elem = map.find(key);
        if (elem == map.end())
            return def;
        // use the , separator
        std::vector<T> res;
        for (auto e : parseAsVector(elem->second, {sep}, false))
        {
            T aux;
            std::stringstream sstr;
            sstr << e;
            sstr >> aux;
            res.push_back(aux);
        }
        if (res.size() < minElms)
            return def;
        return res;
    }

    void saveToFile(const std::string &path)
    {
        onSaveToFile();
        std::ofstream file(path, std::ios::binary);
        if (!file)
            throw std::runtime_error("SafeMap::saveToFile could not open file " + path);
        saveToStream(file);
    }

    void saveToStream(std::ostream &ostr, char kv_sep = ' ', char elem_sep = '\n') const
    {
        for (const auto &kv : map)
            ostr << kv.first << std::string(1, kv_sep) << kv.second << std::string(1, elem_sep);
    }

    void readFromStream(std::istream &istr, const std::set<char> &separators = {' ', '\t', '=', ';', '\n'})
    {
        // make a string
        istr.seekg(std::ios::end);
        std::string str;
        str.reserve(istr.tellg());
        istr.seekg(std::ios::beg);
        char c = istr.get();
        while (!istr.eof())
        {
            str.push_back(c);
            c = istr.get();
        }
        auto velements = parseAsVector(str, separators);
        map.clear();
        if (velements.size() > 1)
        {
            for (size_t i = 0; i < velements.size() - 1; i += 2)
                insert({velements[i], velements[i + 1]});
        }
        onReadedFromFile();
    }

    void readFromFile(const std::string &path, const std::set<char> &separators = {' ', '\t', '=', ';', '\n'})
    {
        std::ifstream file(path);
        if (!file)
            throw std::runtime_error("Invalid file path:" + path);
        readFromStream(file, separators);
    }

    template <typename T> static std::string v2String(const std::vector<T> &vec)
    {
        if (vec.size() == 0)
            return "";
        std::string str;
        for (auto d : vec)
            str += std::to_string(d) + ",";
        return str;
    }

    static ParamSet fromString(const std::string &str, const std::set<char> &separators = {' ', '\t', '=', ';', '\n'})
    {

        auto velements = parseAsVector(str, separators);
        ParamSet data;
        if (velements.size() > 0)
        {
            for (int i = 0; i < int(velements.size()) - 1; i += 2)
            {
                data.insert({velements[i], velements[i + 1]});
            }
        }
        return data;
    }

    static ParamSet fromString(const std::string &str, char sep)
    {

        auto velements = parseAsVector(str, sep);
        ParamSet data;
        if (velements.size() > 0)
        {
            for (int i = 0; i < int(velements.size()) - 1; i += 2)
            {
                data.insert({velements[i], velements[i + 1]});
            }
        }
        return data;
    }

    std::string toRawString() const
    {
        auto raw_string = toString(1, 2);
        for (auto &c : raw_string)
            if (c == '\"')
                c = 3;
        return raw_string;
    }

    static ParamSet fromRawString(const std::string &raw_stringin)
    {
        std::string raw_string = raw_stringin;
        // replace 3 by \"
        for (auto &c : raw_string)
            if (c == 3)
                c = '\"';
        return ParamSet::fromString(raw_string, {1, 2});
    }

    virtual std::string toString(char kv_sep = ' ', char elem_sep = '\n') const
    {
        std::stringstream str;
        saveToStream(str, kv_sep, elem_sep);
        return str.str();
    }
    virtual void onSaveToFile()
    {
    }
    virtual void onReadedFromFile()
    {
    }
    size_t size() const
    {
        return map.size();
    }

    std::string toJson(bool addInitEnd = true)
    {
        std::string res;
        if (addInitEnd)
            res = "{";
        for (auto kv : *this)
            res += "\"" + kv.first + ":\"" + kv.second + "\",";
        if (res.back() == ',')
            res.pop_back();
        if (addInitEnd)
            res += "}";
        return res;
    }

    void toStream(std::ostream &str) const
    {
        auto toStream__ = [](const std::string &m, std::ostream &str) {
            uint32_t s = m.size();
            str.write((char *)&s, sizeof(s));
            str.write(&m[0], m.size());
        };

        toStream__(toRawString(), str);
    }
    void fromStream(std::istream &str)
    {
        auto fromStream__ = [](std::string &m, std::istream &str) {
            uint32_t s = 0;
            str.read((char *)&s, sizeof(s));
            if (str.gcount() != sizeof(s))
                s = 0;
            m.resize(s);
            str.read(&m[0], m.size());
        };
        std::string sstr;
        fromStream__(sstr, str);
        *this = fromRawString(sstr);
    }

  public:
    static std::vector<std::string> parseAsVector(const std::string &str, char sep)
    {
        std::set<char> ssep = {sep};
        return parseAsVector(str, ssep);
    }
    static std::vector<std::string> parseAsVector(const std::string &str,
                                                  const std::set<char> &separators = {' ', '\t', '=', ';', '\n'},
                                                  bool force_even = true)
    {
        std::vector<std::string> elements;
        std::string substr;
        substr.reserve(25);
        bool isString = false;
        for (const char &c : str)
        {
            if (c == '\"')
            {
                if (isString)
                {
                    elements.push_back(substr);
                    isString = false;
                    substr.clear();
                }
                else
                {
                    isString = true;
                }
            }
            else
            {
                if (isString)
                {
                    substr.push_back(c);
                }
                else if (separators.count(c))
                {
                    if (!substr.empty())
                    {
                        elements.push_back(substr);
                        substr.clear();
                    }
                    if (c == ';' || c == '\n')
                    {
                        if (!substr.empty())
                        {
                            elements.push_back(substr);
                            substr.clear();
                        }
                        if (elements.size() % 2 != 0)
                            elements.push_back(""); // add to keep it even
                    }
                }
                else
                    substr.push_back(c);
            }
        }
        if (!substr.empty())
        {
            elements.push_back(substr);
            substr.clear();
        }
        if (elements.size() % 2 != 0 && force_even)
            elements.push_back(""); // add to keep it even
        return elements;
    }

    static bool parseURI(const std::string &url_wp, std::string &host, std::string &port, std::string &path, int defPort = 80)
    {

        // remove params
        std::string url;
        url.reserve(url_wp.size());
        for (auto c : url_wp)
        {
            if (c == '?')
                break;
            else
                url.push_back(c);
        }

        // find first ://
        auto curpos = url.find("://");
        if (curpos == std::string::npos)
            curpos = 0;
        else
            curpos += 3;
        // read until the first /
        auto startPort = url.find(":", curpos);
        auto startPath = url.find("/", curpos);
        if (startPath == std::string::npos)
            startPath = url.size();
        int endHostName = 0;
        if (startPort == std::string::npos)
            endHostName = startPath;
        else
            endHostName = startPort;

        for (int i = curpos; i < endHostName; i++)
            host.push_back(url[i]);

        curpos = endHostName + 1;
        if (startPort == std::string::npos)
        {
            port = std::to_string(defPort);
        }
        else
        {
            for (size_t i = curpos; i < startPath; i++)
                port.push_back(url[i]);
        }
        curpos = startPath;
        for (size_t i = curpos; i < url.size(); i++)
            path.push_back(url[i]);

        return true;
    }

    static std::tuple<std::string, std::string, std::string> parseURI(const std::string &uri, int defPort = 80)
    {
        std::string host, port, address;
        if (!parseURI(uri, host, port, address, defPort))
            return std::tuple<std::string, std::string, std::string>();
        else
            return std::tuple<std::string, std::string, std::string>(host, port, address);
    }
    static ParamSet getURIParams(const std::string &url)
    {
        btwutils::ParamSet pset;
        size_t pos = 0;
        for (pos = 0; pos < url.size(); pos++)
        {
            if (url[pos] == '?')
            {
                pos++;
                break;
            }
        }

        std::string params;
        for (; pos < url.size(); pos++)
            params.push_back(url[pos]);
        if (params.size() > 0)
        {

            for (auto &c : params)
                if (c == '&' || c == '=')
                    c = ' ';
            std::stringstream sstr;
            sstr << params;
            while (!sstr.eof())
            {
                std::string key, val;
                if (sstr >> key >> val)
                    pset[key] = val;
            }
        }
        return pset;
    }
};

} // namespace btwutils
#endif

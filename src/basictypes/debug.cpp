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

#include "debug.h"
#include <fstream>
namespace ucoslam
{
namespace debug
{
int Debug::level = 0;
std::map<std::string, std::string> Debug::strings;
bool Debug::_showTimer = false;
void Debug::showTimer(bool v)
{
    _showTimer = v;
}

void Debug::addString(const std::string &label, const std::string &data)
{
    strings.insert(make_pair(label, data));
}

std::string Debug::getString(const std::string &str)
{
    auto it = strings.find(str);
    if (it == strings.end())
        return "";
    else
        return it->second;
}

bool Debug::isString(const std::string &str)
{
    return strings.count(str) != 0;
}

bool Debug::isInited = false;

void Debug::setLevel(int l)
{
    level = l;
    isInited = false;
    init();
}
int Debug::getLevel()
{
    init();
    return level;
}
void Debug::init()
{
    if (!isInited)
    {
        isInited = true;
        if (level >= 1)
        {
        }
    }
}

} // namespace debug
} // namespace ucoslam

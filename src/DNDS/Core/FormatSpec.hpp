#pragma once

#include "DNDS/Core/Errors.hpp"
#include <fmt/format.h>
#include <string>

namespace DNDS::Format
{
    struct FloatSpec
    {
        char align = '>';
        char sign = ' ';
        int width = -1;
        int precision = 6;
        char type = 'g';

        std::string ToFormatString(bool includeAlignSign = true) const
        {
            if (!includeAlignSign)
            {
                if (width < 0)
                    return fmt::format(FMT_STRING("{{:.{}{}}}"), precision, type);
                return fmt::format(FMT_STRING("{{:{}.{}{}}}"), width, precision, type);
            }
            if (width < 0)
                return fmt::format(FMT_STRING("{{:{0}{1}.{2}{3}}}"), align, sign, precision, type);
            return fmt::format(FMT_STRING("{{:{0}{1}{2}.{3}{4}}}"), align, sign, width, precision, type);
        }
    };

    /// Parse a floating-point format specification `[align][sign][width][.precision][type]`
    /// up to `}` or `end`. Throws on invalid characters when @p strict is true.
    template <class Iter>
    Iter ParseFloatSpec(Iter it, Iter end, FloatSpec &spec, bool strict = true)
    {
        bool afterDot = false;
        while (it != end && *it != '}')
        {
            switch (*it)
            {
            case '<':
            case '>':
            case '^':
                spec.align = *it++;
                break;
            case '+':
            case '-':
            case ' ':
                spec.sign = *it++;
                break;
            case 'e':
            case 'E':
            case 'f':
            case 'F':
            case 'g':
            case 'G':
                spec.type = *it++;
                break;
            case '.':
                afterDot = true;
                ++it;
                break;
            default:
                if (*it >= '0' && *it <= '9')
                {
                    int val = 0;
                    while (it != end && *it >= '0' && *it <= '9')
                    {
                        val = val * 10 + (*it - '0');
                        ++it;
                    }
                    if (afterDot)
                        spec.precision = val;
                    else
                        spec.width = val;
                }
                else if (strict)
                {
                    DNDS_check_throw_info(false, fmt::format("invalid float format spec character '{}'", *it));
                }
                else
                {
                    ++it;
                }
                break;
            }
        }
        return it;
    }
}

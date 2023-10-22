#include "Defines.hpp"

// #ifdef _MSC_VER
// #define _SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING
// #endif
#include <codecvt>

namespace DNDS
{
    std::ostream *logStream;

    bool useCout = true;

    std::ostream &log() { return useCout ? std::cout : *logStream; }

    void setLogStream(std::ostream *nstream) { useCout = false, logStream = nstream; }

    std::string getStringForceWString(const std::wstring &v)
    {
        // std::vector<char> buf(v.size());
        // std::wcstombs(buf.data(), v.data(), v.size());
        // return std::string{buf.data()};
        DISABLE_WARNING_PUSH
        DISABLE_WARNING_DEPRECATED_DECLARATIONS
        std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
        return converter.to_bytes(v); // TODO: on windows use WideCharToMultiByte()
        DISABLE_WARNING_POP
    }
}
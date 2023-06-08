#include "Defines.hpp"

namespace DNDS
{
    class SerializerBase
    {
    public:
        virtual ~SerializerBase() = 0;
        virtual void OpenFile(const std::string &fName) = 0;
        virtual void CloseFile() = 0;
        virtual void CreatePath(const std::string &p) = 0;
        virtual void GoToPath(const std::string &p) = 0;

        virtual void WriteInt(const std::string &name, int v) = 0;
        virtual void WriteIndex(const std::string &name, index v) = 0;
        virtual void WriteReal(const std::string &name, real v) = 0;
        virtual void WriteIndexVector(const std::string &name, const std::vector<index> &v) = 0;
        virtual void WriteRowsizeVector(const std::string &name, const std::vector<rowsize> &v) = 0;
        virtual void WriteRealVector(const std::string &name, const std::vector<real> &v) = 0;
        virtual void WriteString(const std::string &name, const std::string &v) = 0;
        virtual void WriteSharedIndexVector(const std::string &name, const ssp<std::vector<index>> &v) = 0;

        virtual void ReadInt(const std::string &name, int &v) = 0;
        virtual void ReadIndex(const std::string &name, index &v) = 0;
        virtual void ReadReal(const std::string &name, real &v) = 0;
        virtual void ReadIndexVector(const std::string &name, std::vector<index> &v) = 0;
        virtual void ReadRowsizeVector(const std::string &name, std::vector<rowsize> &v) = 0;
        virtual void ReadRealVector(const std::string &name, std::vector<real> &v) = 0;
        virtual void ReadString(const std::string &name, std::string &v) = 0;
        virtual void ReadSharedIndexVector(const std::string &name, ssp<std::vector<index>> &v) = 0;

        
    };

}
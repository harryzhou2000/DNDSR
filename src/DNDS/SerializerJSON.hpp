#pragma once
#include "SerializerBase.hpp"

#include "json.hpp"
#include <fstream>
#include <map>

namespace DNDS
{
    class SerializerJSON : public SerializerBase
    {
        std::fstream fileStream;
        nlohmann::json jObj;
        bool reading = true;
        std::vector<std::string> cPathSplit;
        std::string cP;
        std::map<void *, std::string> ptr_2_pth;
        std::map<std::string, void *> pth_2_ssp;

        bool useCodecOnUint8{false};

    public:
        void SetUseCodecOnUint8(bool v) { useCodecOnUint8 = v; }

        void OpenFile(const std::string &fName, bool read) override;
        void CloseFile() override;
        void CreatePath(const std::string &p) override;
        void GoToPath(const std::string &p) override;
        std::string GetCurrentPath() override;

        void WriteInt(const std::string &name, int v) override;
        void WriteIndex(const std::string &name, index v) override;
        void WriteReal(const std::string &name, real v) override;
        void WriteIndexVector(const std::string &name, const std::vector<index> &v) override;
        void WriteRowsizeVector(const std::string &name, const std::vector<rowsize> &v) override;
        void WriteRealVector(const std::string &name, const std::vector<real> &v) override;
        void WriteString(const std::string &name, const std::string &v) override;
        void WriteSharedIndexVector(const std::string &name, const ssp<std::vector<index>> &v) override;
        void WriteSharedRowsizeVector(const std::string &name, const ssp<std::vector<rowsize>> &v) override;

        void WriteUint8Array(const std::string &name, const uint8_t *data, index size) override;

        void ReadInt(const std::string &name, int &v) override;
        void ReadIndex(const std::string &name, index &v) override;
        void ReadReal(const std::string &name, real &v) override;
        void ReadIndexVector(const std::string &name, std::vector<index> &v) override;
        void ReadRowsizeVector(const std::string &name, std::vector<rowsize> &v) override;
        void ReadRealVector(const std::string &name, std::vector<real> &v) override;
        void ReadString(const std::string &name, std::string &v) override;
        void ReadSharedIndexVector(const std::string &name, ssp<std::vector<index>> &v) override;
        void ReadSharedRowsizeVector(const std::string &name, ssp<std::vector<rowsize>> &v) override;

        void ReadUint8Array(const std::string &name, uint8_t *data, index &size) override;

        ~SerializerJSON() override {}
    };
}
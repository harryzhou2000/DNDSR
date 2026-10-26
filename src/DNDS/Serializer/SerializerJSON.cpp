#include "SerializerJSON.hpp"
#include <algorithm>
#include <string>
#include "base64_rfc4648.hpp"
#include <zlib.h>
#include <fmt/core.h>
#include <set>
#include <climits>

namespace DNDS::Serializer
{

    /**
     * @brief returns good path and if the path is absolute
     */
    bool processPath(std::vector<std::string> &pth)
    {
        bool ifAbs = pth.empty() || pth[0].empty();
        pth.erase(
            std::remove_if(pth.begin(), pth.end(), [](const std::string &v)
                           { return v.empty(); }), // only keep non-zero sized path names
            pth.end());

        return ifAbs;
    }

    std::string constructPath(std::vector<std::string> &pth)
    {
        std::string ret;
        for (auto &name : pth)
            ret.append(std::string("/") + name);
        return ret;
    }

    void SerializerJSON::OpenFile(const std::string &fName, bool read)
    {
        DNDS_assert(!fileStream.is_open());
        reading = read;
        if (read)
            fileStream.open(fName, std::ios::in);
        else
            fileStream.open(fName, std::ios::out);
        DNDS_assert_info(fileStream.is_open(), fmt::format(" attempted to {} file [{}]", read ? "read" : "write", fName));

        if (reading)
            fileStream >> jObj;
        else
            jObj = nlohmann::json::object();
        cP = "";
    }
    void SerializerJSON::CloseFile()
    {
        CloseFileNonVirtual();
    }
    void SerializerJSON::CloseFileNonVirtual()
    {
        // DNDS_assert(fileStream.is_open());
        if (!reading && bool(fileStream))
            fileStream << jObj;
        fileStream.close();
        cPathSplit.clear();
        dedupClear();
        jObj.clear();
    }
    void SerializerJSON::CreatePath(const std::string &p)
    {
        auto pth = splitSString(p, '/');
        // std::cout << pth.size() << std::endl;
        // std::cout << cP + constructPath(pth) << std::endl;
        bool isAbs = processPath(pth);
        if (isAbs)
            jObj[nlohmann::json::json_pointer(constructPath(pth))] = nlohmann::json::object();
        else
            jObj[nlohmann::json::json_pointer(cP + constructPath(pth))] = nlohmann::json::object();
    }
    void SerializerJSON::GoToPath(const std::string &p)
    {
        auto pth = splitSString(p, '/');
        bool isAbs = processPath(pth);
        if (isAbs)
            cPathSplit = std::move(pth);
        else
            for (auto &name : pth)
                cPathSplit.push_back(name);
        cP = constructPath(cPathSplit);
        DNDS_assert(jObj[nlohmann::json::json_pointer(cP)].is_object());
    }
    std::string SerializerJSON::GetCurrentPath()
    {
        return cP;
    }

    std::set<std::string> SerializerJSON::ListCurrentPath()
    {
        auto cPointer = nlohmann::json::json_pointer(cP);
        const auto &v = jObj[cPointer];
        DNDS_assert_info(v.is_object(), fmt::format("current path is not an object " + cP));
        std::set<std::string> ret;
        for (const auto &[key, value] : v.items())
            ret.insert(key);
        return ret;
    }

    void SerializerJSON::WriteInt(const std::string &name, int v)
    {
        auto cPointer = nlohmann::json::json_pointer(cP);
        jObj[cPointer][name] = v;
    }
    void SerializerJSON::WriteIndex(const std::string &name, index v)
    {
        auto cPointer = nlohmann::json::json_pointer(cP);
        jObj[cPointer][name] = v;
    }
    void SerializerJSON::WriteReal(const std::string &name, real v)
    {
        auto cPointer = nlohmann::json::json_pointer(cP);
        jObj[cPointer][name] = v;
    }
    void SerializerJSON::WriteIndexVector(const std::string &name, const std::vector<index> &v, ArrayGlobalOffset offset)
    {
        auto cPointer = nlohmann::json::json_pointer(cP);
        jObj[cPointer][name] = v;
    }
    void SerializerJSON::WriteRowsizeVector(const std::string &name, const std::vector<rowsize> &v, ArrayGlobalOffset offset)
    {
        auto cPointer = nlohmann::json::json_pointer(cP);
        jObj[cPointer][name] = v;
    }
    void SerializerJSON::WriteRealVector(const std::string &name, const std::vector<real> &v, ArrayGlobalOffset offset)
    {
        auto cPointer = nlohmann::json::json_pointer(cP);
        jObj[cPointer][name] = v;
    }
    void SerializerJSON::WriteString(const std::string &name, const std::string &v)
    {
        auto cPointer = nlohmann::json::json_pointer(cP);
        jObj[cPointer][name] = v;
    }
    void SerializerJSON::WriteSharedIndexVector(const std::string &name, const ssp<host_device_vector<index>> &v, ArrayGlobalOffset offset)
    {
        auto cPointer = nlohmann::json::json_pointer(cP);
        std::string refPath;
        if (dedupLookup(v, refPath))
            jObj[cPointer][name]["ref"] = refPath;
        else
        {
            jObj[cPointer][name] = *v;
            dedupRegister(v, cP + "/" + name);
        }
    }
    void SerializerJSON::WriteSharedRowsizeVector(const std::string &name, const ssp<host_device_vector<rowsize>> &v, ArrayGlobalOffset offset)
    {
        auto cPointer = nlohmann::json::json_pointer(cP);
        std::string refPath;
        if (dedupLookup(v, refPath))
            jObj[cPointer][name]["ref"] = refPath;
        else
        {
            jObj[cPointer][name] = *v;
            dedupRegister(v, cP + "/" + name);
        }
    }

    void SerializerJSON::ReadInt(const std::string &name, int &v)
    {
        auto cPointer = nlohmann::json::json_pointer(cP);
        DNDS_assert(jObj[cPointer][name].is_number_integer());
        v = jObj[cPointer][name].get<std::remove_reference_t<decltype(v)>>();
    }
    void SerializerJSON::ReadIndex(const std::string &name, index &v)
    {
        auto cPointer = nlohmann::json::json_pointer(cP);
        DNDS_assert(jObj[cPointer][name].is_number_integer());
        v = jObj[cPointer][name].get<std::remove_reference_t<decltype(v)>>();
    }
    void SerializerJSON::ReadReal(const std::string &name, real &v)
    {
        auto cPointer = nlohmann::json::json_pointer(cP);
        DNDS_assert(jObj[cPointer][name].is_number_float());
        v = jObj[cPointer][name].get<std::remove_reference_t<decltype(v)>>();
    }
    void SerializerJSON::ReadIndexVector(const std::string &name, std::vector<index> &v, ArrayGlobalOffset &offset)
    {
        auto cPointer = nlohmann::json::json_pointer(cP);
        DNDS_assert(jObj[cPointer][name].is_array());
        v = jObj[cPointer][name].get<std::remove_reference_t<decltype(v)>>();
        offset = ArrayGlobalOffset_Unknown;
    }
    void SerializerJSON::ReadRowsizeVector(const std::string &name, std::vector<rowsize> &v, ArrayGlobalOffset &offset)
    {
        auto cPointer = nlohmann::json::json_pointer(cP);
        DNDS_assert(jObj[cPointer][name].is_array());
        v = jObj[nlohmann::json::json_pointer(cP)][name].get<std::remove_reference_t<decltype(v)>>();
        offset = ArrayGlobalOffset_Unknown;
    }
    void SerializerJSON::ReadRealVector(const std::string &name, std::vector<real> &v, ArrayGlobalOffset &offset)
    {
        auto cPointer = nlohmann::json::json_pointer(cP);
        DNDS_assert(jObj[cPointer][name].is_array());
        v = jObj[cPointer][name].get<std::remove_reference_t<decltype(v)>>();
        offset = ArrayGlobalOffset_Unknown;
    }
    void SerializerJSON::ReadString(const std::string &name, std::string &v)
    {
        auto cPointer = nlohmann::json::json_pointer(cP);
        DNDS_assert(jObj[cPointer][name].is_string());
        v = jObj[cPointer][name].get<std::remove_reference_t<decltype(v)>>();
    }
    std::string SerializerJSON::ResolveSharedIndexPath(const std::string &name)
    {
        auto cPointer = nlohmann::json::json_pointer(cP);
        std::string refPath;
        if (jObj[cPointer][name].is_object() && !jObj[cPointer][name].is_array())
        {
            DNDS_assert(jObj[cPointer][name]["ref"].is_string());
            refPath = jObj[cPointer][name]["ref"].get<std::string>();
        }
        else
        {
            refPath = cP + "/" + name;
        }
        return refPath;
    }

    void SerializerJSON::ReadSharedIndexVector(const std::string &name, ssp<host_device_vector<index>> &v, ArrayGlobalOffset &offset)
    {
        using tValue = host_device_vector<index>;
        auto refPath = ResolveSharedIndexPath(name);

        if (!sharedReadLookup(refPath, ArrayGlobalOffset_Unknown, v))
        {
            DNDS_assert(jObj[nlohmann::json::json_pointer(refPath)].is_array());
            // TODO: OPTIMIZE into direct read from json
            auto v_temp = jObj[nlohmann::json::json_pointer(refPath)].get<std::vector<index>>();
            v = std::make_shared<tValue>(v_temp); // vector's copy constructor
            sharedReadRegister(refPath, ArrayGlobalOffset_Unknown, v);
        }
        offset = ArrayGlobalOffset_Unknown;
    }
    void SerializerJSON::WriteSharedRowStartVector(const std::string &name,
                                                   const ssp<const host_device_vector<index>> &v, ArrayGlobalOffset data)
    {
        ValidateRowStartWrite(v, data);
        auto pointer = nlohmann::json::json_pointer(cP);
        auto found = rowStartWrites.find(v.get());
        if (found != rowStartWrites.end())
            jObj[pointer][name]["ref"] = found->second.path;
        else
        {
            jObj[pointer][name] = *v;
            rowStartWrites.emplace(v.get(), RowStartWriteEntry{v, data, v->size(), cP + "/" + name});
        }
    }

    ArrayGlobalOffset SerializerJSON::ReadSharedRowStartVector(const std::string &name,
                                                               ssp<host_device_vector<index>> &v, ArrayGlobalOffset rows)
    {
        auto path = ResolveSharedIndexPath(name);
        ValidateRowStartRead(path, rows);
        const auto &stored = jObj[nlohmann::json::json_pointer(path)];
        RowStartCheck(stored.is_array() && rows.offset() == 0 && stored.size() == size_t(rows.size()) + 1,
                      "JSON row-start reads require the complete local row structure");
        auto found = rowStartReads.find(path);
        if (found == rowStartReads.end())
        {
            auto local = std::make_shared<host_device_vector<index>>(stored.size());
            for (size_t i = 0; i < stored.size(); ++i)
                (*local)[i] = stored[i].get<index>();
            auto data = NormalizeRowStarts(*local, false);
            found = rowStartReads.emplace(path, RowStartReadEntry{rows, data, std::move(local)}).first;
        }
        v = found->second.local;
        return found->second.data;
    }

    void SerializerJSON::ReadSharedRowsizeVector(const std::string &name, ssp<host_device_vector<rowsize>> &v, ArrayGlobalOffset &offset)
    {
        auto cPointer = nlohmann::json::json_pointer(cP);
        using tValue = host_device_vector<rowsize>;
        std::string refPath;
        if (jObj[cPointer][name].is_object() && !jObj[cPointer][name].is_array())
        {
            DNDS_assert(jObj[cPointer][name]["ref"].is_string());
            refPath = jObj[cPointer][name]["ref"].get<std::string>();
        }
        else
        {
            refPath = cP + "/" + name;
        }

        if (!sharedReadLookup(refPath, ArrayGlobalOffset_Unknown, v))
        {
            DNDS_assert(jObj[nlohmann::json::json_pointer(refPath)].is_array());
            // TODO: OPTIMIZE into direct read from json
            auto v_temp = jObj[nlohmann::json::json_pointer(refPath)].get<std::vector<rowsize>>();
            v = std::make_shared<tValue>(v_temp); // vector's copy constructor
            sharedReadRegister(refPath, ArrayGlobalOffset_Unknown, v);
        }
        offset = ArrayGlobalOffset_Unknown;
    }
    void SerializerJSON::WriteByteSegments(const std::string &name, const uint8_t *data, index memoryBytes,
                                           const std::vector<ByteMemorySegment> &segments, ArrayGlobalOffset offset)
    {
        const auto total = ValidateByteSegments(data, memoryBytes, segments);
        DNDS_check_throw_info(!reading && offset == ArrayGlobalOffset_Unknown,
                              "JSON byte segments require a writable per-rank record");
        // The DOM is the encoding buffer. Do not concatenate rows into another
        // byte vector or invoke the legacy whole-buffer compression path.
        auto &stored = jObj[nlohmann::json::json_pointer(cP)][name];
        stored = nlohmann::json::array();
        auto &values = stored.get_ref<nlohmann::json::array_t &>();
        values.reserve(size_t(total));
        for (const auto &segment : segments)
            for (index byte = 0; byte < segment.size; ++byte)
                values.emplace_back(data[segment.offset + byte]);
    }

    index SerializerJSON::IndexVectorSize(const std::string &name)
    {
        const auto &value = jObj.at(nlohmann::json::json_pointer(cP)).at(name);
        DNDS_check_throw_info(value.is_array() && value.size() <= size_t(INT64_MAX), "invalid index metadata extent");
        return index(value.size());
    }

    void SerializerJSON::ReadByteSegments(const std::string &name, uint8_t *data, index memoryBytes,
                                          const std::vector<ByteMemorySegment> &segments, ArrayGlobalOffset &offset)
    {
        const auto total = ValidateByteSegments(data, memoryBytes, segments);
        DNDS_check_throw_info(reading && offset == ArrayGlobalOffset_Unknown,
                              "JSON byte segments require a readable per-rank record");
        const auto &stored = jObj.at(nlohmann::json::json_pointer(cP)).at(name);
        if (stored.is_object())
        {
            // Version-2 checkpoints used a base64/zlib object. Decode its
            // compressed input a base64 word at a time and inflate directly
            // into the final selected rows: no payload-sized decode buffers.
            DNDS_check_throw_info(stored.contains("size") && stored.at("size").is_number_integer() &&
                                      stored.at("size").get<index>() == total && stored.contains("encoded") && stored.at("encoded").is_string(),
                                  "invalid legacy JSON byte encoding");
            const auto &encoded = stored.at("encoded").get_ref<const nlohmann::json::string_t &>();
            DNDS_check_throw_info(encoded.size() % 4 == 0, "invalid legacy base64 length");
            auto digit = [](char c) -> int
            {
                if (c >= 'A' && c <= 'Z')
                    return c - 'A';
                if (c >= 'a' && c <= 'z')
                    return c - 'a' + 26;
                if (c >= '0' && c <= '9')
                    return c - '0' + 52;
                if (c == '+')
                    return 62;
                if (c == '/')
                    return 63;
                return -1;
            };
            z_stream stream{};
            DNDS_check_throw_info(inflateInit(&stream) == Z_OK, "legacy JSON inflate initialization failed");
            struct EndInflate
            {
                z_stream *stream;
                ~EndInflate() { inflateEnd(stream); }
            } end{&stream};
            size_t segment = 0;
            index used = 0, written = 0;
            bool finished = false;
            uint8_t overflow{};
            for (size_t input = 0; input < encoded.size(); input += 4)
            {
                const bool last = input + 4 == encoded.size();
                const int a = digit(encoded[input]), b = digit(encoded[input + 1]);
                const bool pad2 = encoded[input + 2] == '=', pad3 = encoded[input + 3] == '=';
                const int c = pad2 ? 0 : digit(encoded[input + 2]), d = pad3 ? 0 : digit(encoded[input + 3]);
                DNDS_check_throw_info(a >= 0 && b >= 0 && c >= 0 && d >= 0 && (!pad2 || pad3) &&
                                          (!(pad2 || pad3) || last) && (!pad2 || (b & 15) == 0) && (!pad3 || pad2 || (c & 3) == 0) && !finished,
                                      "invalid or trailing legacy base64 input");
                const uint32_t word = (uint32_t(a) << 18) | (uint32_t(b) << 12) | (uint32_t(c) << 6) | uint32_t(d);
                uint8_t compressed[3]{uint8_t(word >> 16), uint8_t(word >> 8), uint8_t(word)};
                stream.next_in = compressed;
                stream.avail_in = pad2 ? 1 : pad3 ? 2
                                                  : 3;
                do
                {
                    while (segment < segments.size() && used == segments[segment].size)
                    {
                        ++segment;
                        used = 0;
                    }
                    const bool hasOutput = segment < segments.size();
                    const auto capacity = hasOutput ? uInt(std::min<index>(segments[segment].size - used, UINT_MAX)) : uInt(1);
                    stream.next_out = hasOutput ? data + segments[segment].offset + used : &overflow;
                    stream.avail_out = capacity;
                    const auto before = stream.avail_in;
                    const int result = inflate(&stream, Z_NO_FLUSH);
                    const auto produced = capacity - stream.avail_out;
                    if (result == Z_BUF_ERROR && stream.avail_in == 0 && produced == 0)
                        break;
                    DNDS_check_throw_info((result == Z_OK || result == Z_STREAM_END) && (hasOutput || produced == 0),
                                          "legacy JSON inflate failed or expanded beyond its declared size");
                    used += produced;
                    written += produced;
                    finished = result == Z_STREAM_END;
                    DNDS_check_throw_info(!finished || (last && stream.avail_in == 0 && written == total),
                                          "legacy JSON compressed payload has trailing data or wrong size");
                    DNDS_check_throw_info(finished || produced || stream.avail_in != before, "legacy JSON inflate made no progress");
                } while ((stream.avail_in || stream.avail_out == 0) && !finished);
            }
            DNDS_check_throw_info(finished && written == total, "truncated legacy JSON compressed payload");
            return;
        }
        DNDS_check_throw_info(stored.is_array() && stored.size() == size_t(total),
                              "JSON byte segment payload has an incompatible encoding or size");
        // Validate the entire encoding before modifying caller memory.
        for (const auto &value : stored)
            DNDS_check_throw_info(value.is_number_integer() &&
                                      (value.is_number_unsigned() ? value.get<uint64_t>() <= 255
                                                                  : value.get<int64_t>() >= 0 && value.get<int64_t>() <= 255),
                                  "JSON byte segment payload contains an invalid byte");
        size_t source = 0;
        for (const auto &segment : segments)
            for (index byte = 0; byte < segment.size; ++byte)
                data[segment.offset + byte] = stored[source++].get<uint8_t>();
    }

    void SerializerJSON::WriteUint8Array(const std::string &name, const uint8_t *data, index size, ArrayGlobalOffset offset)
    {
        auto zlibCompressedSize = [&](index sizeC)
        {
            return sizeC + (sizeC + 999) / 1000 + 12; // form vtk
        };
        int compressLevel = this->deflateLevel;
        auto zlibCompressData = [&](const uint8_t *buf, index sizeC)
        {
            std::vector<uint8_t> ret(zlibCompressedSize(sizeC));
            uLongf retSize = ret.size();
            auto ret_v = compress2(ret.data(), &retSize, buf, sizeC, compressLevel);
            if (ret_v != Z_OK)
                DNDS_assert_info(false, "compression failed");
            ret.resize(retSize);
            return ret;
        };

        auto cPointer = nlohmann::json::json_pointer(cP);
        if (!useCodecOnUint8)
            jObj[cPointer][name] = std::vector<uint8_t>(data, data + size);
        else
        {
            jObj[cPointer][name]["size"] = size;
            jObj[cPointer][name]["encoded"] = cppcodec::base64_rfc4648::encode(
                zlibCompressData(data, size));
        }
    }
    void SerializerJSON::ReadUint8Array(const std::string &name, uint8_t *data, index &size, ArrayGlobalOffset &offset)
    {
        auto cPointer = nlohmann::json::json_pointer(cP);
        DNDS_assert(jObj[cPointer][name].is_object() || jObj[cPointer][name].is_array());
        if (jObj[cPointer][name].is_array())
        {
            size = jObj[cPointer][name].size();
            if (data != nullptr)
                std::copy(jObj[cPointer][name].begin(), jObj[cPointer][name].end(), data);
        }
        else if (jObj[cPointer][name].is_object() &&
                 jObj[cPointer][name]["size"].is_number_integer() &&
                 jObj[cPointer][name]["encoded"].is_string())
        {
            size = jObj[cPointer][name]["size"];
            auto &dataIn = jObj[cPointer][name]["encoded"].get_ref<nlohmann::json::string_t &>();
            if (data != nullptr)
            {
                std::vector<uint8_t> decodedData;
                cppcodec::base64_rfc4648::decode(decodedData, dataIn);
                std::vector<uint8_t> decodedDataUncompress(size);
                uLongf sizeU{(uLongf)size};
                auto ret = uncompress(decodedDataUncompress.data(), &sizeU, decodedData.data(), decodedData.size());
                DNDS_assert_info(ret == Z_OK, "zlib uncompress failed");
                DNDS_assert_info(sizeU == size, "zlib uncompress failed");
                // DNDS_assert(decodedData.size() == size);
                std::copy(decodedDataUncompress.begin(), decodedDataUncompress.end(), data);
            }
        }
        else
        {
            DNDS_assert(false);
        }
        offset = ArrayGlobalOffset_Unknown;
    }
} // namespace DNDS::Serializer

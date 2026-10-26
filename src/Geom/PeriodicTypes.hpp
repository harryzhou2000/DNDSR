#pragma once

#include "Geometric.hpp"
#include "BoundaryCondition.hpp"
#include "DNDS/Serializer/SerializerBase.hpp"
#include "DNDS/Runtime/ElementSchema.hpp"

namespace DNDS::Geom
{
    struct NodePeriodicBits
    {
        uint8_t _v{0U};
        [[nodiscard]] bool getP1() const { return _v & 0x01U; }
        [[nodiscard]] bool getP2() const { return _v & 0x02U; }
        [[nodiscard]] bool getP3() const { return _v & 0x04U; }
        void setP1True() { _v |= 0x01U; }
        void setP2True() { _v |= 0x02U; }
        void setP3True() { _v |= 0x04U; }
        DNDS_DEVICE_TRIVIAL_COPY_DEFINE(NodePeriodicBits, NodePeriodicBits)
        DNDS_DEVICE_CALLABLE uint8_t operator^(const NodePeriodicBits &r) const
        {
            return uint8_t(_v ^ r._v);
        }
        DNDS_DEVICE_CALLABLE NodePeriodicBits operator&(const NodePeriodicBits &r) const
        {
            return NodePeriodicBits{uint8_t(_v & r._v)};
        }
        DNDS_DEVICE_CALLABLE operator uint8_t() const
        {
            return uint8_t{_v};
        }
        DNDS_DEVICE_CALLABLE operator bool() const
        {
            return bool(_v);
        }
        DNDS_DEVICE_CALLABLE bool operator==(const NodePeriodicBits &r) const
        {
            return uint8_t(r) == uint8_t(*this);
        }
        static MPI_Datatype CommType() { return MPI_UINT8_T; }
        static int CommMult() { return 1; }

        friend std::ostream &operator<<(std::ostream &o, const NodePeriodicBits &b)
        {
            o << int(b._v);
            return o;
        }
    };

    static const NodePeriodicBits nodePB1{0x01U};
    static const NodePeriodicBits nodePB2{0x02U};
    static const NodePeriodicBits nodePB3{0x04U};

    struct NodeIndexPBI
    {
        index i{UnInitIndex};
        NodePeriodicBits pbi{};

        bool operator<(const NodeIndexPBI &r) const
        {
            if (i < r.i)
                return true;
            else if (i == r.i)
                return uint8_t(pbi) < uint8_t(r.pbi);
            else
                return false;
        }

        bool operator==(const NodeIndexPBI &r) const
        {
            return r.i == i && r.pbi == pbi;
        }

        bool operator!=(const NodeIndexPBI &r) const { return !(*this == r); }
    };

    inline bool isCollaborativeNodePeriodicBits(const std::vector<NodePeriodicBits> &a, const std::vector<NodePeriodicBits> &b)
    {
        size_t n = a.size();
        DNDS_assert(n == b.size());
        if (n == 0)
            return true;
        auto v0 = a[0] ^ b[0];
        for (size_t i = 1; i < n; i++)
            if ((a.at(i) ^ b.at(i)) != v0)
                return false;
        return true;
    }

    struct Periodicity
    {
        std::array<tGPointPortable, 4> rotation{};
        std::array<tPointPortable, 4> translation{};
        std::array<tPointPortable, 4> rotationCenter{};
        DNDS_DEVICE_TRIVIAL_COPY_DEFINE_NO_EMPTY_CTOR(Periodicity, Periodicity)
        DNDS_DEVICE_CALLABLE Periodicity()
        {
            for (auto &r : rotation)
                r.map().setIdentity();
            for (auto &r : rotationCenter)
                r.map().setZero();
            translation[0].map().setZero();
            translation[1].map() = tPoint{1, 0, 0};
            translation[2].map() = tPoint{0, 1, 0};
            translation[3].map() = tPoint{0, 0, 1};

            //     translation[1] = tPoint{0, 0, 0};
            //     rotation[1] << 0, 1, 0,
            //         -1, 0, 0,
            //         0, 0, 1;
        }

        DNDS_HOST void WriteSerializer(Serializer::SerializerBaseSSP serializerP, const std::string &name)
        {
            auto cwd = serializerP->GetCurrentPath();
            serializerP->CreatePath(name);
            serializerP->GoToPath(name);

            for (int i = 1; i <= 3; i++)
            {
                serializerP->WriteRealVector("rotation" + std::to_string(i), Geom::JacobiToSTDVector(rotation.at(i).map()), Serializer::ArrayGlobalOffset_One);
                serializerP->WriteRealVector("rotationCenter" + std::to_string(i), Geom::VectorToSTDVector(rotationCenter.at(i).map()), Serializer::ArrayGlobalOffset_One);
                serializerP->WriteRealVector("translation" + std::to_string(i), Geom::VectorToSTDVector(translation.at(i).map()), Serializer::ArrayGlobalOffset_One);
            }

            serializerP->GoToPath(cwd);
        }

        DNDS_HOST void ReadSerializer(Serializer::SerializerBaseSSP serializerP, const std::string &name)
        {
            auto cwd = serializerP->GetCurrentPath();
            // serializerP->CreatePath(name); // * no create
            serializerP->GoToPath(name);

            for (int i = 1; i <= 3; i++)
            {
                std::vector<real> rotRead, rotCRead, transRead;
                Serializer::ArrayGlobalOffset offsetV = Serializer::ArrayGlobalOffset_One; // must explicitly set as One
                serializerP->ReadRealVector("rotation" + std::to_string(i), rotRead, offsetV);
                serializerP->ReadRealVector("rotationCenter" + std::to_string(i), rotCRead, offsetV);
                serializerP->ReadRealVector("translation" + std::to_string(i), transRead, offsetV);
                rotation.at(i).map() = Geom::STDVectorToJacobi(rotRead);
                rotationCenter.at(i).map() = Geom::STDVectorToVector(rotCRead);
                translation.at(i).map() = Geom::STDVectorToVector(transRead);
            }

            serializerP->GoToPath(cwd);
        }

        DNDS_DEVICE_CALLABLE [[nodiscard]] tPoint TransCoord(const tPoint &c, t_index id) const
        {
            DNDS_assert(FaceIDIsPeriodic(id));
            t_index i{0};
            if (FaceIDIsPeriodicDonor(id))
                i = -id - 3;
            else
                i = -id;
            return rotation.at(i).map() * (c - rotationCenter.at(i).map()) + rotationCenter.at(i).map() + translation.at(i).map();
        }

        DNDS_DEVICE_CALLABLE [[nodiscard]] tPoint TransCoordBack(const tPoint &c, t_index id) const
        {
            DNDS_assert(FaceIDIsPeriodic(id));
            t_index i{0};
            if (FaceIDIsPeriodicDonor(id))
                i = -id - 3;
            else
                i = -id;
            return rotation.at(i).map().transpose() * ((c - translation.at(i).map()) - rotationCenter.at(i).map()) + rotationCenter.at(i).map();
        }

        ///@todo //TODO: add support for cartesian tensor transformation

        template <int dim, int nVec>
        DNDS_DEVICE_CALLABLE Eigen::Matrix<real, dim, nVec> TransVector(const Eigen::Matrix<real, dim, nVec> &v, t_index id) const
        {
            DNDS_assert(FaceIDIsPeriodic(id));
            t_index i{0};
            if (FaceIDIsPeriodicDonor(id))
                i = -id - 3;
            else
                i = -id;
            if constexpr (dim == 3)
                return rotation.at(i).map() * v;
            else
                return rotation.at(i).map()({0, 1}, {0, 1}) * v;
        }

        template <int dim, int nVec>
        DNDS_DEVICE_CALLABLE Eigen::Matrix<real, dim, nVec> TransVectorBack(const Eigen::Matrix<real, dim, nVec> &v, t_index id) const
        {
            DNDS_assert(FaceIDIsPeriodic(id));
            t_index i{0};
            if (FaceIDIsPeriodicDonor(id))
                i = -id - 3;
            else
                i = -id;
            if constexpr (dim == 3)
                return rotation.at(i).map().transpose() * v;
            else
                return rotation.at(i).map()({0, 1}, {0, 1}).transpose() * v;
        }

        template <int dim>
        DNDS_DEVICE_CALLABLE Eigen::Matrix<real, dim, dim> TransMat(const Eigen::Matrix<real, dim, dim> &m, t_index id) const
        {
            DNDS_assert(FaceIDIsPeriodic(id));
            t_index i{0};
            if (FaceIDIsPeriodicDonor(id))
                i = -id - 3;
            else
                i = -id;
            if constexpr (dim == 3)
                return rotation.at(i).map() * m * rotation.at(i).map().transpose();
            else
                return rotation.at(i).map()({0, 1}, {0, 1}) * m * rotation.at(i).map()({0, 1}, {0, 1}).transpose();
        }

        template <int dim>
        DNDS_DEVICE_CALLABLE Eigen::Matrix<real, dim, dim> TransMatBack(const Eigen::Matrix<real, dim, dim> &m, t_index id) const
        {
            DNDS_assert(FaceIDIsPeriodic(id));
            t_index i{0};
            if (FaceIDIsPeriodicDonor(id))
                i = -id - 3;
            else
                i = -id;
            if constexpr (dim == 3)
                return rotation.at(i).map().transpose() * m * rotation.at(i).map();
            else
                return rotation.at(i).map()({0, 1}, {0, 1}).transpose() * m * rotation.at(i).map()({0, 1}, {0, 1});
        }

        DNDS_DEVICE_CALLABLE [[nodiscard]] tPoint GetCoordByBits(const tPoint &c, const NodePeriodicBits &bits) const
        {
            if (!bool(bits))
                return c;
            tPoint ret = c;
            if (bits.getP3())
                ret = this->TransCoord(ret, BC_ID_PERIODIC_3);
            if (bits.getP2())
                ret = this->TransCoord(ret, BC_ID_PERIODIC_2);
            if (bits.getP1())
                ret = this->TransCoord(ret, BC_ID_PERIODIC_1);
            return ret;
        }

        template <int dim, int nVec>
        DNDS_DEVICE_CALLABLE auto GetVectorByBits(const Eigen::Matrix<real, dim, nVec> &v, const NodePeriodicBits &bits) const
        {
            if (!bool(bits))
                return v;
            Eigen::Matrix<real, dim, nVec> ret = v;
            if (bits.getP3())
                ret = this->TransVector<dim, nVec>(ret, BC_ID_PERIODIC_3);
            if (bits.getP2())
                ret = this->TransVector<dim, nVec>(ret, BC_ID_PERIODIC_2);
            if (bits.getP1())
                ret = this->TransVector<dim, nVec>(ret, BC_ID_PERIODIC_1);
            return ret;
        }

        DNDS_DEVICE_CALLABLE [[nodiscard]] tPoint GetCoordBackByBits(const tPoint &c, const NodePeriodicBits &bits) const
        {
            if (!bool(bits))
                return c;
            tPoint ret = c;
            if (bits.getP1())
                ret = this->TransCoordBack(ret, BC_ID_PERIODIC_1);
            if (bits.getP2())
                ret = this->TransCoordBack(ret, BC_ID_PERIODIC_2);
            if (bits.getP3())
                ret = this->TransCoordBack(ret, BC_ID_PERIODIC_3);
            return ret;
        }

        template <int dim, int nVec>
        DNDS_DEVICE_CALLABLE auto GetVectorBackByBits(const Eigen::Matrix<real, dim, nVec> &v, const NodePeriodicBits &bits) const
        {
            if (!bool(bits))
                return v;
            Eigen::Matrix<real, dim, nVec> ret = v;
            if (bits.getP1())
                ret = this->TransVectorBack<dim, nVec>(ret, BC_ID_PERIODIC_1);
            if (bits.getP2())
                ret = this->TransVectorBack<dim, nVec>(ret, BC_ID_PERIODIC_2);
            if (bits.getP3())
                ret = this->TransVectorBack<dim, nVec>(ret, BC_ID_PERIODIC_3);
            return ret;
        }
    };
} // namespace DNDS::Geom

namespace DNDS::Runtime
{
    template <>
    struct ElementSchemaProvider<Geom::NodePeriodicBits>
    {
        static ElementSchema Get()
        {
            return ElementSchema::Record("Geom.NodePeriodicBits.v1", sizeof(Geom::NodePeriodicBits),
                                         {{"_v", offsetof(Geom::NodePeriodicBits, _v), ElementScalar::U8}},
                                         "DNDS::Geom::NodePeriodicBits", alignof(Geom::NodePeriodicBits));
        }
    };
}

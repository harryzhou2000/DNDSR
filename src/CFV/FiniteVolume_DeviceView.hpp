#pragma once
#include "FiniteVolumeSettings.hpp"
#include "VRDefines.hpp"
#include "Geom/DiffTensors.hpp"
#include "Geom/Mesh/MeshView.hpp"

namespace DNDS::CFV
{
    template <DeviceBackend B, class MeshView = Geom::UnstructuredMeshView>
    class FiniteVolumeDeviceView
    {
    public:
        MeshView mesh;
        Geom::Periodicity meshPeriodicity;

        FiniteVolumeSettings settings;

        static_assert(std::is_trivially_copyable_v<FiniteVolumeSettings>);
        static_assert(std::is_trivially_copyable_v<MeshView>);

        DNDS_DEVICE_CALLABLE int MeshDimension() const
        {
            if constexpr (std::is_same_v<MeshView, Geom::UnstructuredMeshView>)
                return mesh.Dimension();
            else
                return mesh.dim;
        }
        DNDS_DEVICE_CALLABLE bool MeshIsPeriodic() const
        {
            if constexpr (std::is_same_v<MeshView, Geom::UnstructuredMeshView>)
                return mesh.IsPeriodic();
            else
                return mesh.isPeriodic;
        }
        DNDS_DEVICE_CALLABLE static const Geom::Periodicity &ReadMeshPeriodicity(const MeshView &source)
        {
            if constexpr (std::is_same_v<MeshView, Geom::UnstructuredMeshView>)
                return source.Periodic();
            else
                return source.periodicInfo;
        }
        DNDS_DEVICE_CALLABLE Geom::Periodicity &MeshPeriodicity() { return meshPeriodicity; }

        DNDS_DEVICE_CALLABLE int getDim()
        {
            return MeshDimension();
        }

    protected:
        real volGlobal{0};
        tScalarPair::t_deviceView<B> volumeLocal;
        tScalarPair::t_deviceView<B> faceArea;
        tRecAtrPair::t_deviceView<B> cellAtr;
        tRecAtrPair::t_deviceView<B> faceAtr;
        tCoeffPair::t_deviceView<B> cellIntJacobiDet;
        tCoeffPair::t_deviceView<B> faceIntJacobiDet;
        t3VecsPair::t_deviceView<B> faceUnitNorm;
        t3VecPair::t_deviceView<B> faceMeanNorm;
        t3VecPair::t_deviceView<B> cellBary;
        t3VecPair::t_deviceView<B> faceCent;
        t3VecPair::t_deviceView<B> cellCent;
        t3VecsPair::t_deviceView<B> cellIntPPhysics;
        t3VecsPair::t_deviceView<B> faceIntPPhysics;
        t3VecPair::t_deviceView<B> cellAlignedHBox;
        t3VecPair::t_deviceView<B> cellMajorHBox;
        t3MatPair::t_deviceView<B> cellMajorCoord;
        t3MatPair::t_deviceView<B> cellInertia;
        tScalarPair::t_deviceView<B> cellSmoothScale;

        static_assert(std::is_trivially_copyable_v<tScalarPair::t_deviceView<B>>);
        static_assert(std::is_trivially_copyable_v<tRecAtrPair::t_deviceView<B>>);
        static_assert(std::is_trivially_copyable_v<tCoeffPair::t_deviceView<B>>);
        static_assert(std::is_trivially_copyable_v<t3VecsPair::t_deviceView<B>>);
        static_assert(std::is_trivially_copyable_v<t3VecPair::t_deviceView<B>>);
        static_assert(std::is_trivially_copyable_v<t3MatPair::t_deviceView<B>>);

    public:
        DNDS_DEVICE_TRIVIAL_COPY_DEFINE_NO_EMPTY_CTOR(FiniteVolumeDeviceView, FiniteVolumeDeviceView)

        template <class TMain>
        FiniteVolumeDeviceView(TMain &fv, index placeholder, MeshView meshView)
            : mesh(meshView), meshPeriodicity(ReadMeshPeriodicity(meshView)), settings(fv.getDim())
        {
            DNDS_assert(placeholder == UnInitIndex);
            DNDS_COPY_MEMBER(fv, settings);

            DNDS_COPY_MEMBER(fv, volGlobal);

            DNDS_COPY_MEMBER_VIEW(fv, volumeLocal);
            DNDS_COPY_MEMBER_VIEW(fv, faceArea);
            DNDS_COPY_MEMBER_VIEW(fv, cellAtr);
            DNDS_COPY_MEMBER_VIEW(fv, faceAtr);
            DNDS_COPY_MEMBER_VIEW(fv, cellIntJacobiDet);
            DNDS_COPY_MEMBER_VIEW(fv, faceIntJacobiDet);
            DNDS_COPY_MEMBER_VIEW(fv, faceUnitNorm);
            DNDS_COPY_MEMBER_VIEW(fv, faceMeanNorm);
            DNDS_COPY_MEMBER_VIEW(fv, cellBary);
            DNDS_COPY_MEMBER_VIEW(fv, faceCent);
            DNDS_COPY_MEMBER_VIEW(fv, cellCent);
            DNDS_COPY_MEMBER_VIEW(fv, cellIntPPhysics);
            DNDS_COPY_MEMBER_VIEW(fv, faceIntPPhysics);
            DNDS_COPY_MEMBER_VIEW(fv, cellAlignedHBox);
            DNDS_COPY_MEMBER_VIEW(fv, cellMajorHBox);
            DNDS_COPY_MEMBER_VIEW(fv, cellMajorCoord);
            DNDS_COPY_MEMBER_VIEW(fv, cellInertia);
            DNDS_COPY_MEMBER_VIEW(fv, cellSmoothScale);
        }

        DNDS_DEVICE_CALLABLE real GetCellVol(index iCell) { return volumeLocal[iCell][0]; }
        DNDS_DEVICE_CALLABLE real GetFaceArea(index iFace) { return faceArea[iFace][0]; }
        DNDS_DEVICE_CALLABLE real GetCellSmoothScaleRatio(index iCell) const
        {
            return cellSmoothScale(iCell, 0);
        }
        DNDS_DEVICE_CALLABLE real GetGlobalVol() const { return volGlobal; }

        DNDS_DEVICE_CALLABLE real GetCellJacobiDet(index iCell, rowsize iG) const { return cellIntJacobiDet(iCell, iG); }
        DNDS_DEVICE_CALLABLE real GetFaceJacobiDet(index iFace, rowsize iG) const { return faceIntJacobiDet(iFace, iG); }

        DNDS_DEVICE_CALLABLE Geom::Elem::Quadrature GetFaceQuad(index iFace) const
        {
            auto e = mesh.GetFaceElement(iFace);
            return Geom::Elem::Quadrature{e, faceAtr(iFace, 0).intOrder};
        }

        DNDS_DEVICE_CALLABLE Geom::Elem::Quadrature GetFaceQuadO1(index iFace) const
        {
            auto e = mesh.GetFaceElement(iFace);
            return Geom::Elem::Quadrature{e, 1};
        }

        DNDS_DEVICE_CALLABLE real GetFaceParamArea(index iFace) const { return std::get<1>(this->GetFaceQuadO1(iFace).GetQuadraturePointInfo(0)); }

        DNDS_DEVICE_CALLABLE Geom::Elem::Quadrature GetCellQuad(index iCell) const
        {
            auto e = mesh.GetCellElement(iCell);
            return Geom::Elem::Quadrature{e, cellAtr(iCell, 0).intOrder};
        }

        DNDS_DEVICE_CALLABLE Geom::Elem::Quadrature GetCellQuadO1(index iCell) const
        {
            auto e = mesh.GetCellElement(iCell);
            return Geom::Elem::Quadrature{e, 1};
        }

        DNDS_DEVICE_CALLABLE real GetCellParamVol(index iCell) const { return std::get<1>(this->GetCellQuadO1(iCell).GetQuadraturePointInfo(0)); }

        DNDS_DEVICE_CALLABLE Geom::tPoint GetCellBary(index iCell) { return cellBary[iCell]; }

        DNDS_DEVICE_CALLABLE bool CellIsFaceBack(index iCell, index iFace, rowsize ic2f) const
        {
            return mesh.CellIsFaceBack(iCell, iFace, ic2f);
        }

        DNDS_DEVICE_CALLABLE index CellFaceOther(index iCell, index iFace, rowsize ic2f) const
        {
            return mesh.CellFaceOther(iCell, iFace, ic2f);
        }

        DNDS_DEVICE_CALLABLE Geom::tPoint GetFaceNorm(index iFace, int iG) const
        {
            if (iG >= 0)
                return faceUnitNorm(iFace, iG);
            else
                return faceMeanNorm[iFace];
        }

        DNDS_DEVICE_CALLABLE Geom::tPoint GetFaceNormFromCell(index iFace, index iCell, rowsize if2c, int iG)
        {
            if (!MeshIsPeriodic())
                return GetFaceNorm(iFace, iG);
            auto faceID = mesh.GetFaceZone(iFace);
            if (!Geom::FaceIDIsPeriodic(faceID))
                return GetFaceNorm(iFace, iG);
            DNDS_assert(if2c >= 0);
            if (if2c == 1 && Geom::FaceIDIsPeriodicMain(faceID))
                return MeshPeriodicity().TransVector(GetFaceNorm(iFace, iG), faceID);
            if (if2c == 1 && Geom::FaceIDIsPeriodicDonor(faceID))
                return MeshPeriodicity().TransVectorBack(GetFaceNorm(iFace, iG), faceID);
            return GetFaceNorm(iFace, iG);
        }

        DNDS_DEVICE_CALLABLE Geom::tPoint GetFaceQuadraturePPhys(index iFace, int iG)
        {
            if (iG >= 0)
                return faceIntPPhysics(iFace, iG);
            else
                return faceCent[iFace];
        }

        DNDS_DEVICE_CALLABLE Geom::tPoint GetFaceQuadraturePPhysFromCell(index iFace, index iCell, rowsize if2c, int iG)
        {
            if (!MeshIsPeriodic())
                return GetFaceQuadraturePPhys(iFace, iG);
            auto faceID = mesh.GetFaceZone(iFace);
            if (!Geom::FaceIDIsPeriodic(faceID))
                return GetFaceQuadraturePPhys(iFace, iG);
            DNDS_assert(if2c >= 0);
            if (if2c == 1 && Geom ::FaceIDIsPeriodicMain(faceID)) // I am donor
            {
                // std::cout << iFace <<" " << iCell << " " <<if2c << std::endl;
                // std::cout << GetFaceQuadraturePPhys(iFace, iG).transpose() << std::endl;
                // std::cout << mesh->periodicInfo.TransCoord(GetFaceQuadraturePPhys(iFace, iG), faceID).transpose() << std::endl;
                // std::abort();
                return MeshPeriodicity().TransCoord(GetFaceQuadraturePPhys(iFace, iG), faceID);
            }
            if (if2c == 1 && Geom::FaceIDIsPeriodicDonor(faceID)) // I am main
                return MeshPeriodicity().TransCoordBack(GetFaceQuadraturePPhys(iFace, iG), faceID);
            return GetFaceQuadraturePPhys(iFace, iG);
        }

        DNDS_DEVICE_CALLABLE Geom::tPoint GetFacePointFromCell(index iFace, index iCell, rowsize if2c, const Geom::tPoint &pnt)
        {
            if (!MeshIsPeriodic())
                return pnt;
            auto faceID = mesh.GetFaceZone(iFace);
            if (!Geom::FaceIDIsPeriodic(faceID))
                return pnt;
            DNDS_assert(if2c >= 0);
            if (if2c == 1 && Geom ::FaceIDIsPeriodicMain(faceID)) // I am donor
            {
                // std::cout << iFace <<" " << iCell << " " <<if2c << std::endl;
                // std::cout << pnt.transpose() << std::endl;
                // std::cout << mesh->periodicInfo.TransCoord(pnt, faceID).transpose() << std::endl;
                // std::abort();
                return MeshPeriodicity().TransCoord(pnt, faceID);
            }
            if (if2c == 1 && Geom::FaceIDIsPeriodicDonor(faceID)) // I am main
                return MeshPeriodicity().TransCoordBack(pnt, faceID);
            return pnt;
        }

        DNDS_DEVICE_CALLABLE Geom::tPoint GetOtherCellBaryFromCell(
            index iCell, index iCellOther,
            index iFace, rowsize if2c)
        {
            if (!MeshIsPeriodic())
                return GetCellBary(iCellOther);

            auto faceID = mesh.GetFaceZone(iFace);
            if (!Geom::FaceIDIsPeriodic(faceID))
                return GetCellBary(iCellOther);
            DNDS_assert(if2c >= 0);
            if ((if2c == 1 && Geom::FaceIDIsPeriodicMain(faceID)) ||
                (if2c == 0 && Geom::FaceIDIsPeriodicDonor(faceID))) // I am donor
                return MeshPeriodicity().TransCoord(GetCellBary(iCellOther), faceID);
            if ((if2c == 1 && Geom::FaceIDIsPeriodicDonor(faceID)) ||
                (if2c == 0 && Geom::FaceIDIsPeriodicMain(faceID))) // I am main
                return MeshPeriodicity().TransCoordBack(GetCellBary(iCellOther), faceID);
            return GetCellBary(iCellOther);
        }

        DNDS_DEVICE_CALLABLE Geom::tPoint GetOtherCellPointFromCell(
            index iCell, index iCellOther,
            index iFace, rowsize if2c, const Geom::tPoint &pnt)
        {
            if (!MeshIsPeriodic())
                return pnt;

            auto faceID = mesh.GetFaceZone(iFace);
            if (!Geom::FaceIDIsPeriodic(faceID))
                return pnt;
            DNDS_assert(if2c >= 0);
            if ((if2c == 1 && Geom::FaceIDIsPeriodicMain(faceID)) ||
                (if2c == 0 && Geom::FaceIDIsPeriodicDonor(faceID))) // I am donor
                return MeshPeriodicity().TransCoord(pnt, faceID);
            if ((if2c == 1 && Geom::FaceIDIsPeriodicDonor(faceID)) ||
                (if2c == 0 && Geom::FaceIDIsPeriodicMain(faceID))) // I am main
                return MeshPeriodicity().TransCoordBack(pnt, faceID);
            return pnt;
        }

        DNDS_DEVICE_CALLABLE Geom::tGPoint GetOtherCellInertiaFromCell(
            index iCell, index iCellOther,
            index iFace, rowsize if2c)
        { // structure copy of GetOtherCellBaryFromCell
            if (!MeshIsPeriodic())
                return cellInertia[iCellOther];

            auto faceID = mesh.GetFaceZone(iFace);
            if (!Geom::FaceIDIsPeriodic(faceID))
                return cellInertia[iCellOther];
            DNDS_assert(if2c >= 0);
            if ((if2c == 1 && Geom::FaceIDIsPeriodicMain(faceID)) ||
                (if2c == 0 && Geom::FaceIDIsPeriodicDonor(faceID))) // I am donor
                return MeshPeriodicity().template TransMat<3>(cellInertia[iCellOther], faceID);
            if ((if2c == 1 && Geom::FaceIDIsPeriodicDonor(faceID)) ||
                (if2c == 0 && Geom::FaceIDIsPeriodicMain(faceID))) // I am main
                return MeshPeriodicity().template TransMatBack<3>(cellInertia[iCellOther], faceID);
            return cellInertia[iCellOther];
        }

        DNDS_DEVICE_CALLABLE Geom::tPoint GetCellQuadraturePPhys(index iCell, int iG)
        {
            if (iG >= 0)
                return cellIntPPhysics(iCell, iG);
            else
                return cellCent[iCell];
        }

        DNDS_DEVICE_CALLABLE real GetCellMaxLenScale(index iCell)
        {
            auto hb = cellMajorHBox[iCell];
            real ret = std::max(hb[0], hb[1]);
            if (this->getDim() == 3)
                ret = std::max(ret, hb[2]);
            return ret * 2;
        }
    };
}

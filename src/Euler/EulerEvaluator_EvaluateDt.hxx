#pragma once
#include "EulerEvaluator.hpp"
#include <CGAL/Simple_cartesian.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_triangle_primitive.h>

namespace DNDS::Euler
{

    template <EulerModel model>
    void EulerEvaluator<model>::GetWallDist()
    {
        using TriArray = ArrayEigenMatrix<3, 3>;
        ssp<TriArray> TrianglesLocal, TrianglesFull;
        DNDS_MAKE_SSP(TrianglesLocal, mesh->getMPI());
        DNDS_MAKE_SSP(TrianglesFull, mesh->getMPI());
        std::vector<Eigen::Matrix<real, 3, 3>> Triangles;
        for (index iBnd = 0; iBnd < mesh->NumBnd(); iBnd++)
        {
            if (pBCHandler->GetTypeFromID(mesh->GetBndZone(iBnd)) == EulerBCType::BCWall)
            {
                index iFace = mesh->bnd2face[iBnd];
                auto elem = mesh->GetFaceElement(iFace);
                if (elem.type == Geom::Elem::ElemType::Line2)
                {
                    Eigen::Matrix<real, 3, 3> tri;
                    tri(Eigen::all, 0) = mesh->coords[mesh->face2node[iFace][0]];
                    tri(Eigen::all, 1) = mesh->coords[mesh->face2node[iFace][1]];
                    tri(Eigen::all, 2) = mesh->coords[mesh->face2node[iFace][1]] + Geom::tPoint{0., 0., vfv->GetFaceArea(iFace)};
                    Triangles.push_back(tri);
                }
                else if (elem.type == Geom::Elem::ElemType::Tri3)
                {
                    Eigen::Matrix<real, 3, 3> tri;
                    tri(Eigen::all, 0) = mesh->coords[mesh->face2node[iFace][0]];
                    tri(Eigen::all, 1) = mesh->coords[mesh->face2node[iFace][1]];
                    tri(Eigen::all, 2) = mesh->coords[mesh->face2node[iFace][2]];
                    Triangles.push_back(tri);
                }
                else if (elem.type == Geom::Elem::ElemType::Quad4)
                {
                    Eigen::Matrix<real, 3, 3> tri;
                    tri(Eigen::all, 0) = mesh->coords[mesh->face2node[iFace][0]];
                    tri(Eigen::all, 1) = mesh->coords[mesh->face2node[iFace][1]];
                    tri(Eigen::all, 2) = mesh->coords[mesh->face2node[iFace][2]];
                    Triangles.push_back(tri);
                    tri(Eigen::all, 0) = mesh->coords[mesh->face2node[iFace][0]];
                    tri(Eigen::all, 1) = mesh->coords[mesh->face2node[iFace][2]];
                    tri(Eigen::all, 2) = mesh->coords[mesh->face2node[iFace][3]];
                    Triangles.push_back(tri);
                }
                else
                {
                    DNDS_assert_info(false, "This elem not implemented yet for GetWallDist()");
                }
            }
        }
        TrianglesLocal->Resize(Triangles.size(), 3, 3);
        for (index i = 0; i < TrianglesLocal->Size(); i++)
            (*TrianglesLocal)[i] = Triangles[i];
        Triangles.clear();
        ArrayTransformerType<TriArray>::Type TrianglesTransformer;
        TrianglesTransformer.setFatherSon(TrianglesLocal, TrianglesFull);
        TrianglesTransformer.createFatherGlobalMapping();

        std::vector<index> pullingSet;
        pullingSet.resize(TrianglesTransformer.pLGlobalMapping->globalSize());
        for (index i = 0; i < pullingSet.size(); i++)
            pullingSet[i] = i;
        TrianglesTransformer.createGhostMapping(pullingSet);
        TrianglesTransformer.createMPITypes();
        TrianglesTransformer.pullOnce();
        std::cout << "To search in " << TrianglesFull->Size() << std::endl;

        typedef CGAL::Simple_cartesian<double> K;
        typedef K::FT FT;
        // typedef K::Ray_3 Ray;
        // typedef K::Line_3 Line;
        typedef K::Point_3 Point;
        typedef K::Triangle_3 Triangle;
        typedef std::vector<Triangle>::iterator Iterator;
        typedef CGAL::AABB_triangle_primitive<K, Iterator> Primitive;
        typedef CGAL::AABB_traits<K, Primitive> AABB_triangle_traits;
        typedef CGAL::AABB_tree<AABB_triangle_traits> Tree;

        std::vector<Triangle> triangles;
        triangles.reserve(TrianglesFull->Size());

        for (index i = 0; i < TrianglesFull->Size(); i++)
        {
            Point p0((*TrianglesFull)[i](0, 0), (*TrianglesFull)[i](1, 0), (*TrianglesFull)[i](2, 0));
            Point p1((*TrianglesFull)[i](0, 1), (*TrianglesFull)[i](1, 1), (*TrianglesFull)[i](2, 1));
            Point p2((*TrianglesFull)[i](0, 2), (*TrianglesFull)[i](1, 2), (*TrianglesFull)[i](2, 2));
            triangles.push_back(Triangle(p0, p1, p2));
        }
        TrianglesLocal->Resize(0, 3, 3);
        TrianglesFull->Resize(0, 3, 3);
        double minDist = veryLargeReal;
        this->dWall.resize(mesh->NumCellProc());

        if (!triangles.empty())
        {
            // std::cout << "tree building" << std::endl;
            Tree tree(triangles.begin(), triangles.end());

            // std::cout << "tree built" << std::endl;
            // search

            for (index iCell = 0; iCell < mesh->NumCellProc(); iCell++)
            {
                // std::cout << "iCell " << iCell << std::endl;
                auto quadCell = vfv->GetCellQuad(iCell);
                dWall[iCell].resize(quadCell.GetNumPoints());
                for (int ig = 0; ig < quadCell.GetNumPoints(); ig++)
                {
                    // std::cout << "iG " << ig << std::endl;
                    auto p = vfv->GetCellQuadraturePPhys(iCell, ig);
                    Point pQ(p[0], p[1], p[2]);
                    // std::cout << "pQ " << pQ << std::endl;
                    // Point closest_point = tree.closest_point(pQ);
                    FT sqd = tree.squared_distance(pQ);
                    // std::cout << "sqd" << sqd << std::endl;
                    dWall[iCell][ig] = std::max(std::sqrt(sqd), 1e-12);
                    if (dWall[iCell][ig] < minDist)
                        minDist = dWall[iCell][ig];
                }
            }
        }
        else
        {
            for (index iCell = 0; iCell < mesh->NumCellProc(); iCell++)
            {
                // std::cout << "iCell " << iCell << std::endl;
                auto quadCell = vfv->GetCellQuad(iCell);
                dWall[iCell].resize(quadCell.GetNumPoints(), std::pow(veryLargeReal, 1. / 4.));
            }
        }
        std::cout << "MinDist: " << minDist << std::endl;
    }

    // Eigen::Vector<real, -1> EulerEvaluator::CompressRecPart(
    //     const Eigen::Vector<real, -1> &umean,
    //     const Eigen::Vector<real, -1> &uRecInc)1

    //! evaluates dt and facial spectral radius
    template <EulerModel model>
    void EulerEvaluator<model>::EvaluateDt(
        std::vector<real> &dt,
        ArrayDOFV<nVars_Fixed> &u,
        ArrayRECV<nVars_Fixed> &uRec,
        real CFL, real &dtMinall, real MaxDt,
        bool UseLocaldt)
    {
        DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
        InsertCheck(u.father->getMPI(), "EvaluateDt 1");
        for (auto &i : lambdaCell)
            i = 0.0;

        for (index iFace = 0; iFace < mesh->NumFaceProc(); iFace++)
        {
            auto f2c = mesh->face2cell[iFace];
            TVec unitNorm = vfv->GetFaceNorm(iFace, -1)(Seq012);

            index iCellL = f2c[0];
            TU UL = u[iCellL];
            this->UFromCell2Face(UL, iFace, f2c[0], 0);
            TU uMean = UL;
            real pL, asqrL, HL, pR, asqrR, HR;
            TVec vL = UL(Seq123) / UL(0);
            TVec vR = vL;
            Gas::IdealGasThermal(UL(I4), UL(0), vL.squaredNorm(),
                                 settings.idealGasProperty.gamma,
                                 pL, asqrL, HL);
            pR = pL, HR = HL, asqrR = asqrL;
            if (f2c[1] != UnInitIndex)
            {
                TU UR = u[f2c[1]];
                this->UFromCell2Face(UR, iFace, f2c[1], 1);
                uMean = (uMean + UR) * 0.5;
                vR = UR(Seq123) / UR(0);
                Gas::IdealGasThermal(UR(I4), UR(0), vR.squaredNorm(),
                                     settings.idealGasProperty.gamma,
                                     pR, asqrR, HR);
            }
            TDiffU GradULxy, GradURxy;
            GradULxy.resize(Eigen::NoChange, nVars);
            GradURxy.resize(Eigen::NoChange, nVars);
            GradULxy.setZero(), GradURxy.setZero();
            if constexpr (gDim == 2)
                GradULxy({0, 1}, Eigen::all) =
                    vfv->GetIntPointDiffBaseValue(f2c[0], iFace, 0, -1, std::array<int, 2>{1, 2}, 3) *
                    uRec[f2c[0]]; // 2d here
            else
                GradULxy({0, 1, 2}, Eigen::all) =
                    vfv->GetIntPointDiffBaseValue(f2c[0], iFace, 0, -1, std::array<int, 3>{1, 2, 3}, 4) *
                    uRec[f2c[0]]; // 3d here
            this->DiffUFromCell2Face(GradULxy, iFace, f2c[0], 0);
            GradURxy = GradULxy;
            if (f2c[1] != UnInitIndex)
            {
                if constexpr (gDim == 2)
                    GradURxy({0, 1}, Eigen::all) =
                        vfv->GetIntPointDiffBaseValue(f2c[1], iFace, 1, -1, std::array<int, 2>{1, 2}, 3) *
                        uRec[f2c[1]]; // 2d here
                else
                    GradURxy({0, 1, 2}, Eigen::all) =
                        vfv->GetIntPointDiffBaseValue(f2c[1], iFace, 1, -1, std::array<int, 3>{1, 2, 3}, 4) *
                        uRec[f2c[1]]; // 3d here
                this->DiffUFromCell2Face(GradURxy, iFace, f2c[1], 1);
            }
            TDiffU GradUMeanXY = (GradURxy + GradULxy) / 2;

            DNDS_assert(uMean(0) > 0);
            TVec veloMean = (uMean(Seq123).array() / uMean(0)).matrix();
            // real veloNMean = veloMean.dot(unitNorm); // original
            real veloNMean = 0.5 * (vL + vR).dot(unitNorm); // paper
            real veloNL = vL.dot(unitNorm);
            real veloNR = vR.dot(unitNorm);

            // real ekFixRatio = 0.001;
            // Eigen::Vector3d velo = uMean({1, 2, 3}) / uMean(0);
            // real vsqr = velo.squaredNorm();
            // real Ek = vsqr * 0.5 * uMean(0);
            // real Efix = Ek * ekFixRatio;
            // real e = uMean(4) - Ek;
            // if (e < 0)
            //     e = 0.5 * Efix;
            // else if (e < Efix)
            //     e = (e * e + Efix * Efix) / (2 * Efix);
            // uMean(4) = Ek + e;

            real pMean, asqrMean, HMean;
            Gas::IdealGasThermal(uMean(I4), uMean(0), veloMean.squaredNorm(),
                                 settings.idealGasProperty.gamma,
                                 pMean, asqrMean, HMean);

            pMean = (pL + pR) * 0.5;
            real aMean = sqrt(settings.idealGasProperty.gamma * pMean / uMean(0)); // paper

            // DNDS_assert(asqrMean >= 0);
            // real aMean = std::sqrt(asqrMean); // original
            real lambdaConvection = std::abs(veloNMean) + aMean;
            lambdaConvection = std::max(std::sqrt(asqrL) + std::abs(veloNL), std::sqrt(asqrR) + std::abs(veloNR));
            DNDS_assert_info(
                asqrL >= 0 && asqrR >= 0,
                fmt::format(" mean value violates PP! asqr: [{} {}]", asqrL, asqrR)
            );

            // ! refvalue:
            real muRef = settings.idealGasProperty.muGas;

            real gamma = settings.idealGasProperty.gamma;
            real T = pMean / ((gamma - 1) / gamma * settings.idealGasProperty.CpGas * uMean(0));
            real muf = settings.idealGasProperty.muGas *
                       std::pow(T / settings.idealGasProperty.TRef, 1.5) *
                       (settings.idealGasProperty.TRef + settings.idealGasProperty.CSutherland) /
                       (T + settings.idealGasProperty.CSutherland);
            if constexpr (model == NS_SA || model == NS_SA_3D)
            {
                real cnu1 = 7.1;
                real Chi = uMean(I4 + 1) * muRef / muf;
#ifdef USE_NS_SA_NEGATIVE_MODEL
                if (Chi < 10)
                    Chi = 0.05 * std::log(1 + std::exp(20 * Chi));
#endif
                real Chi3 = std::pow(Chi, 3);
                real fnu1 = Chi3 / (Chi3 + std::pow(cnu1, 3));
                muf *= std::max((1 + Chi * fnu1), 1.0);
            }
            if constexpr (model == NS_2EQ || model == NS_2EQ_3D)
            {
                real mut = RANS::GetMut_RealizableKe<dim>(uMean, GradUMeanXY, muf);
                muf = muf + mut;
            }
            real lamVis = muf / uMean(0) *
                          std::max(4. / 3., gamma / settings.idealGasProperty.prGas);

            real lamFace = lambdaConvection * vfv->GetFaceArea(iFace);

            real area = vfv->GetFaceArea(iFace);
            real areaSqr = area * area;
            real volR = vfv->GetCellVol(iCellL);
            // lambdaCell[iCellL] += lamFace + 2 * lamVis * areaSqr / fv->GetCellVol(iCellL);
            if (f2c[1] != UnInitIndex) // can't be non local
                                       // lambdaCell[f2c[1]] += lamFace + 2 * lamVis * areaSqr / fv->volumeLocal[f2c[1]],
                volR = vfv->GetCellVol(f2c[1]);

            lambdaFace[iFace] = lambdaConvection + lamVis * area * (1. / vfv->GetCellVol(iCellL) + 1. / volR);
            lambdaFaceC[iFace] = std::abs(veloNMean) + lamVis * area * (1. / vfv->GetCellVol(iCellL) + 1. / volR); // passive part
            lambdaFaceVis[iFace] = lamVis * area * (1. / vfv->GetCellVol(iCellL) + 1. / volR);

            // if (f2c[0] == 10756)
            // {
            //     std::cout << "----Lambdas" << std::setprecision(16) << iFace << std::endl;
            //     std::cout << lambdaConvection << std::endl;
            //     std::cout << lambdaFaceVis[iFace] << std::endl;
            //     std::cout << veloNMean << " " << aMean << std::endl;
            //     std::cout << gamma << " " << pMean << " " << uMean(0) << std::endl;
            // }

            lambdaCell[iCellL] += lambdaFace[iFace] * vfv->GetFaceArea(iFace);
            if (f2c[1] != UnInitIndex) // can't be non local
                lambdaCell[f2c[1]] += lambdaFace[iFace] * vfv->GetFaceArea(iFace);

            deltaLambdaFace[iFace] = std::abs((vR - vL).dot(unitNorm)) + std::sqrt(std::abs(asqrR - asqrL)) * 0.7071;
        }
        real dtMin = veryLargeReal;
        for (index iCell = 0; iCell < mesh->NumCell(); iCell++)
        {
            // std::cout << fv->GetCellVol(iCell) << " " << (lambdaCell[iCell]) << " " << CFL << std::endl;
            // exit(0);
            dt[iCell] = std::min(CFL * vfv->GetCellVol(iCell) / (lambdaCell[iCell] + 1e-100), MaxDt);
            dtMin = std::min(dtMin, dt[iCell]);
            // if (iCell == 10756)
            // {
            //     std::cout << std::endl;
            // }
        }

        MPI::Allreduce(&dtMin, &dtMinall, 1, DNDS_MPI_REAL, MPI_MIN, u.father->getMPI().comm);

        // if (uRec.father->getMPI().rank == 0)
        //     std::cout << "dt min is " << dtMinall << std::endl;
        if (!UseLocaldt)
        {
            for (auto &i : dt)
                i = dtMinall;
        }
        // if (uRec.father->getMPI().rank == 0)
        // log() << "dt: " << dtMin << std::endl;
    }

}
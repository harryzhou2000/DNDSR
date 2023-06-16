#pragma once
#include "EulerEvaluator.hpp"

namespace DNDS::Euler
{
    // Eigen::Vector<real, -1> EulerEvaluator::CompressRecPart(
    //     const Eigen::Vector<real, -1> &umean,
    //     const Eigen::Vector<real, -1> &uRecInc)

    //! evaluates dt and facial spectral radius
    template <EulerModel model>
    void EulerEvaluator<model>::EvaluateDt(
        std::vector<real> &dt,
        ArrayDOFV<nVars_Fixed> &u,
        real CFL, real &dtMinall, real MaxDt,
        bool UseLocaldt)
    {
        DNDS_FV_EULEREVALUATOR_GET_FIXED_EIGEN_SEQS
        InsertCheck(u.father->mpi, "EvaluateDt 1");
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
           
            DNDS_assert(uMean(0) > 0);
            TVec veloMean = (uMean(Seq123).array() / uMean(0)).matrix();
            // real veloNMean = veloMean.dot(unitNorm); // original
            real veloNMean = 0.5 * (vL + vR).dot(unitNorm); // paper

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

            // ! refvalue:
            real muRef = settings.idealGasProperty.muGas;

            real gamma = settings.idealGasProperty.gamma;
            real T = pMean / ((gamma - 1) / gamma * settings.idealGasProperty.CpGas * uMean(0));
            real muf = settings.idealGasProperty.muGas *
                       std::pow(T / settings.idealGasProperty.TRef, 1.5) *
                       (settings.idealGasProperty.TRef + settings.idealGasProperty.CSutherland) /
                       (T + settings.idealGasProperty.CSutherland);
            if constexpr (model == NS_SA)
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

        MPI_Allreduce(&dtMin, &dtMinall, 1, DNDS_MPI_REAL, MPI_MIN, u.father->mpi.comm);

        // if (uRec.father->mpi.rank == 0)
        //     std::cout << "dt min is " << dtMinall << std::endl;
        if (!UseLocaldt)
        {
            for (auto &i : dt)
                i = dtMinall;
        }
        // if (uRec.father->mpi.rank == 0)
        // log() << "dt: " << dtMin << std::endl;
    }

}
#pragma once
#include "DNDS/Defines.hpp"

namespace DNDS::Euler
{
    template <int nVars_Fixed>
    class ArrayDOFV : public CFV::tUDof<nVars_Fixed>
    {
    public:
        using t_self = ArrayDOFV<nVars_Fixed>;
        void setConstant(real R)
        {
            for (index i = 0; i < this->Size(); i++)
                this->operator[](i).setConstant(R);
        }
        template <class TR>
        void setConstant(const TR &R)
        {
            for (index i = 0; i < this->Size(); i++)
                this->operator[](i) = R;
        }
        void operator+=(t_self &R)
        {
            for (index i = 0; i < this->Size(); i++)
                this->operator[](i) += R.operator[](i);
        }
        void operator-=(t_self &R)
        {
            for (index i = 0; i < this->Size(); i++)
                this->operator[](i) -= R.operator[](i);
        }
        void operator*=(real R)
        {
            for (index i = 0; i < this->Size(); i++)
                this->operator[](i) *= R;
        }
        void operator=(t_self &R)
        {
            for (index i = 0; i < this->Size(); i++)
                this->operator[](i) = R.operator[](i);
        }

        void addTo(t_self &R, real r)
        {
            for (index i = 0; i < this->Size(); i++)
                this->operator[](i) += R.operator[](i) * r;
        }

        void operator*=(std::vector<real> &R)
        {
            DNDS_assert(R.size() >= this->father->Size());
            for (index i = 0; i < this->father->Size(); i++)
                this->operator[](i) *= R[i];
        }

        void operator+=(const Eigen::Vector<real, nVars_Fixed> &R)
        {
            for (index i = 0; i < this->Size(); i++)
                this->operator[](i) += R;
        }

        void operator+=(real R)
        {
            for (index i = 0; i < this->Size(); i++)
                this->operator[](i).array() += R;
        }

        void operator*=(const Eigen::Vector<real, nVars_Fixed> &R)
        {
            for (index i = 0; i < this->Size(); i++)
                this->operator[](i).array() *= R.array();
        }

        void operator*=(t_self &R)
        {
            for (index i = 0; i < this->Size(); i++)
                this->operator[](i).array() *= R.operator[](i).array();
        }

        void operator/=(t_self &R)
        {
            for (index i = 0; i < this->Size(); i++)
                this->operator[](i).array() /= R.operator[](i).array();
        }

        void setAbs()
        {
            for (index i = 0; i < this->Size(); i++)
                this->operator[](i).array() = this->operator[](i).array().abs();
        }

        Eigen::Vector<real, nVars_Fixed> normInc()
        {
            Eigen::Vector<real, nVars_Fixed> ret, retAll;
            ret.resize(this->RowSize());
            retAll.resize(this->RowSize());
            ret.setZero();
            for (index i = 0; i < this->father->Size(); i++) //*note that only father is included
                ret += this->operator[](i).array().abs();
            MPI::Allreduce(ret.data(), retAll.data(), this->RowSize(), DNDS_MPI_REAL, MPI_SUM, this->father->mpi.comm);
            return retAll;
        }

        real norm2()
        {
            real sqrSum{0}, sqrSumAll{0};
            for (index i = 0; i < this->father->Size(); i++) //*note that only father is included
                sqrSum += this->operator[](i).squaredNorm();
            MPI::Allreduce(&sqrSum, &sqrSumAll, 1, DNDS_MPI_REAL, MPI_SUM, this->father->mpi.comm);
            // std::cout << "norm2is " << std::scientific << sqrSumAll << std::endl;
            return std::sqrt(sqrSumAll);
        }

        Eigen::Vector<real, nVars_Fixed> min()
        {
            Eigen::Vector<real, nVars_Fixed> minLocal, min;
            minLocal.resize(this->RowSize());
            minLocal.setConstant(veryLargeReal);
            min = minLocal;
            for (index i = 0; i < this->father->Size(); i++) //*note that only father is included
                minLocal = minLocal.array().min(this->operator[](i).array());
            MPI::Allreduce(&minLocal.data(), &min.data(), minLocal.size(), DNDS_MPI_REAL, MPI_MIN, this->father->mpi.comm);
            return min;
        }

        real dot(const t_self &R)
        {
            real sqrSum{0}, sqrSumAll;
            for (index i = 0; i < this->father->Size(); i++) //*note that only father is included
                sqrSum += this->operator[](i).dot(R.operator[](i));
            MPI::Allreduce(&sqrSum, &sqrSumAll, 1, DNDS_MPI_REAL, MPI_SUM, this->father->mpi.comm);
            return sqrSumAll;
        }
    };

    ///@todo://TODO add operators
    template <int nVars_Fixed>
    class ArrayRECV : public CFV::tURec<nVars_Fixed>
    {
    public:
        using t_self = ArrayRECV<nVars_Fixed>;
        void setConstant(real R)
        {
            for (index i = 0; i < this->Size(); i++)
                this->operator[](i).setConstant(R);
        }
        template <class TR>
        void setConstant(const TR &R)
        {
            for (index i = 0; i < this->Size(); i++)
                this->operator[](i) = R;
        }
        void operator+=(t_self &R)
        {
            for (index i = 0; i < this->Size(); i++)
                this->operator[](i) += R.operator[](i);
        }
        void operator-=(t_self &R)
        {
            for (index i = 0; i < this->Size(); i++)
                this->operator[](i) -= R.operator[](i);
        }
        void operator*=(real R)
        {
            for (index i = 0; i < this->Size(); i++)
                this->operator[](i) *= R;
        }
        void operator*=(std::vector<real> &R)
        {
            DNDS_assert(R.size() >= this->father->Size());
            for (index i = 0; i < this->father->Size(); i++)
                this->operator[](i) *= R[i];
        }
        void operator=(t_self &R)
        {
            for (index i = 0; i < this->Size(); i++)
                this->operator[](i) = R.operator[](i);
        }

        void addTo(t_self &R, real r)
        {
            for (index i = 0; i < this->Size(); i++)
                this->operator[](i) += R.operator[](i) * r;
        }

        real norm2()
        {
            real sqrSum{0}, sqrSumAll{0};
            for (index i = 0; i < this->father->Size(); i++) //*note that only father is included
                sqrSum += this->operator[](i).squaredNorm();
            MPI::Allreduce(&sqrSum, &sqrSumAll, 1, DNDS_MPI_REAL, MPI_SUM, this->father->mpi.comm);
            // std::cout << "norm2is " << std::scientific << sqrSumAll << std::endl;
            return std::sqrt(sqrSumAll);
        }

        real dot(const t_self &R)
        {
            real sqrSum{0}, sqrSumAll;
            for (index i = 0; i < this->father->Size(); i++) //*note that only father is included
                sqrSum += (this->operator[](i).array() * R.operator[](i).array()).sum();
            MPI::Allreduce(&sqrSum, &sqrSumAll, 1, DNDS_MPI_REAL, MPI_SUM, this->father->mpi.comm);
            return sqrSumAll;
        }
    };

    template <int nVars_Fixed>
    class JacobianValue
    {
    public:
        enum Type
        {
            Diagonal = 0,
            DiagonalBlock = 1,
            Full = 2,
        };
        ArrayDOFV<nVars_Fixed> diag, diagInv;
        ArrayEigenMatrix<nVars_Fixed, nVars_Fixed> diagBlock, diagBlockInv;
        ArrayRECV<nVars_Fixed> offDiagBlock;

        void SetDiagonal(ArrayDOFV<nVars_Fixed> &uDof)
        {
            type = Diagonal;
            // todo ! allocate square blocks!
        }

        void SetDiagonalBlock(ArrayDOFV<nVars_Fixed> &uDof)
        {
            type = DiagonalBlock;
            // todo ! allocate square blocks!
        }

        void SetFull(ArrayDOFV<nVars_Fixed> &uDof, Geom::tAdjPair &cell2cell)
        {
            type = Full;
            // todo ! allocate with adjacency!
        }

        void InverseDiag()
        {
            // todo get inverse!
        }

    private:
        Type type = Diagonal;
    };

    enum EulerModel
    {
        NS = 0,
        NS_SA = 1,
        NS_2D = 2,
        NS_3D = 3,
        NS_SA_3D = 4,
        NS_2EQ = 5,
        NS_2EQ_3D = 6,
    };

    constexpr static inline int getNVars_Fixed(const EulerModel model)
    {
        if (model == NS || model == NS_3D)
            return 5;
        else if (model == NS_SA)
            return 6;
        else if (model == NS_SA_3D)
            return 6;
        else if (model == NS_2D)
            return 4;
        else if (model == NS_2EQ || model == NS_2EQ_3D)
            return 7;
        return Eigen::Dynamic;
    }

    constexpr static inline int getNVars(EulerModel model)
    {
        int nVars = getNVars_Fixed(model);
        if (nVars < 0)
        {
            if (model == NS || model == NS_3D)
                return 5;
            else if (model == NS_SA)
                return 6;
            else if (model == NS_SA_3D)
                return 6;
            else if (model == NS_2D)
                return 4;
            else if (model == NS_2EQ || model == NS_2EQ_3D)
                return 7;
            // *** handle variable nVars
        }
        return nVars;
    }

    constexpr static inline int getDim_Fixed(const EulerModel model)
    {
        if (model == NS || model == NS_3D)
            return 3;
        else if (model == NS_SA)
            return 3;
        else if (model == NS_SA_3D)
            return 3;
        else if (model == NS_2D)
            return 2;
        else if (model == NS_2EQ || model == NS_2EQ_3D)
            return 3;
        return Eigen::Dynamic;
    }

    constexpr static inline int getGeomDim_Fixed(const EulerModel model)
    {
        if (model == NS)
            return 2;
        else if (model == NS_SA)
            return 2;
        else if (model == NS_2D)
            return 2;
        else if (model == NS_3D)
            return 3;
        else if (model == NS_SA_3D)
            return 3;
        else if (model == NS_2EQ)
            return 2;
        else if (model == NS_2EQ_3D)
            return 3;
        return Eigen::Dynamic;
    }

    // constexpr static inline bool ifFixedNvars(EulerModel model)
    // {
    //     return (
    //         model == NS ||
    //         model == NS_SA);
    // } // use +/- is ok

    template <int nvars_Fixed, int mul>
    constexpr static inline int nvarsFixedMultiply()
    {
        return nvars_Fixed != Eigen::Dynamic ? nvars_Fixed * mul : Eigen::Dynamic;
    }
}
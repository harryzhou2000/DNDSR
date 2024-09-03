#include "ParametricBaseFunction.hpp"

namespace DNDS::CFV
{

    DNDS_SWITCH_INTELLISENSE(template <int dim>, static const int dim = 3;)
    ParametricBaseCache<dim>::ParametricBaseCache()
    {
        using namespace Geom;
        using namespace Geom::Elem;
        std::set<ElemType> coveredElems;
        coveredElems.insert(Line2);
        coveredElems.insert(Line3);
        coveredElems.insert(Tri3);
        coveredElems.insert(Tri6);
        coveredElems.insert(Quad4);
        coveredElems.insert(Quad9);
        coveredElems.insert(Hex8);
        coveredElems.insert(Hex27);
        coveredElems.insert(Tet4);
        coveredElems.insert(Tet10);
        coveredElems.insert(Prism6);
        coveredElems.insert(Prism18);
        coveredElems.insert(Pyramid5);
        coveredElems.insert(Pyramid14);

        for (int intOrder = 1; intOrder <= INT_ORDER_MAX; intOrder++)
            for (ElemType eType : coveredElems)
            {
                auto elem = Element{eType};
                auto quad = Quadrature(elem, intOrder);
                SummationNoOp noOp;

                {
                    auto &cVec = cache[intOrder][eType][0];
                    cVec.reserve(27);
                    quad.Integration(
                        noOp,
                        [&](SummationNoOp inc, int iG, const tPoint &pParam, const tD01Nj &D01Nj)
                        {
                            cVec.emplace_back();
                            cVec.back().pParam = pParam;
                            tPoint pParamBase = cVec.back().pParam; //! todo: use better origin and scale?
                            cVec.back().DiBjParam.resize(
                                dim == 3 ? ndiffSiz : ndiffSiz2D,
                                dim == 3 ? ndiffSiz : ndiffSiz2D);
                            if constexpr (dim == 3)
                                FPolynomialFill3D(
                                    cVec.back().DiBjParam,
                                    pParamBase(0), pParamBase(1), pParamBase(2),
                                    1, 1, 1,
                                    cVec.back().DiBjParam.rows(),
                                    cVec.back().DiBjParam.cols());
                            if constexpr (dim == 2)
                                FPolynomialFill2D(
                                    cVec.back().DiBjParam,
                                    pParamBase(0), pParamBase(1), pParamBase(2),
                                    1, 1, 1,
                                    cVec.back().DiBjParam.rows(),
                                    cVec.back().DiBjParam.cols());
                        });
                }

                SmallCoordsAsVector cellCoordsParam = GetStandardCoord(elem.type);

                for (int ic2f = 0; ic2f < elem.GetNumFaces(); ic2f++)
                {
                    auto eFace = elem.ObtainFace(ic2f);
                    SmallCoordsAsVector faceCoordsParam;
                    faceCoordsParam.resize(Eigen::NoChange, eFace.GetNumNodes());
                    elem.ExtractFaceNodes(ic2f, cellCoordsParam, faceCoordsParam);
                    auto qFace = Quadrature(eFace, intOrder);

                    auto &cVec = cache[intOrder][eType][1 + ic2f];
                    cVec.reserve(9);
                    qFace.Integration(
                        noOp,
                        [&](SummationNoOp inc, int iG, const tPoint &pParam, const tD01Nj &D01Nj)
                        {
                            cVec.emplace_back();
                            cVec.back().pParam = faceCoordsParam * D01Nj(0, Eigen::all).transpose();
                            tPoint pParamBase = cVec.back().pParam; //! todo: use better origin and scale?
                            cVec.back().DiBjParam.resize(
                                dim == 3 ? ndiffSiz : ndiffSiz2D,
                                dim == 3 ? ndiffSiz : ndiffSiz2D);
                            if constexpr (dim == 3)
                                FPolynomialFill3D(
                                    cVec.back().DiBjParam,
                                    pParamBase(0), pParamBase(1), pParamBase(2),
                                    1, 1, 1,
                                    cVec.back().DiBjParam.rows(),
                                    cVec.back().DiBjParam.cols());
                            if constexpr (dim == 2)
                                FPolynomialFill2D(
                                    cVec.back().DiBjParam,
                                    pParamBase(0), pParamBase(1), pParamBase(2),
                                    1, 1, 1,
                                    cVec.back().DiBjParam.rows(),
                                    cVec.back().DiBjParam.cols());
                            // if (elem.type == Quad4)
                            // {
                            //     std::cout << "Quad4 " << pParamBase.transpose() << std::endl;
                            // }
                        });
                }
            }
    }

    template ParametricBaseCache<2>::ParametricBaseCache();
    template ParametricBaseCache<3>::ParametricBaseCache();
}
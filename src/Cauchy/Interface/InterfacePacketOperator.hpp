#pragma once
#include "Cauchy/Interface/InterfacePacketView.hpp"
#include "DNDS/Runtime/Array/Array.hpp"
#include "DNDS/Runtime/Execution/EagerEngine.hpp"
#include "DNDS/Runtime/Storage/Metadata.hpp"

namespace DNDS::Cauchy
{
    /** Prepared receiver packet assembly. Source grouping and destination
     * uniqueness are validated once at route preparation. CPU accepts multiple
     * donor arrays; CUDA currently requires one complete AoS donor array.
     * The controller orders interpolation/transfers before Submit and publishes
     * the packet after completion. This operator performs neither operation.
     */
    class InterfacePacketOperator
    {
        struct Data;
        std::shared_ptr<const Data> _data;

    public:
        InterfacePacketOperator(const Runtime::Array<InterfacePacketRow> &hostRows,
                                std::vector<Runtime::MetadataBuffer<index>> hostComponents,
                                Runtime::MetadataBuffer<index> hostVectorStarts,
                                index components, index quadratureRows, index faceRows,
                                Runtime::ExecutionContext);
        /// Inputs and output allocations, mapping rows and immutable metadata
        /// are retained through queued CPU/device execution. Other packet rows
        /// are preserved; several connections may fill one packet on one lane.
        void Submit(Runtime::EagerEngine &, const Runtime::ExecutionStream &,
                    const std::vector<Runtime::Array<real>> &values,
                    Runtime::Array<real> &quadrature, Runtime::Array<real> &faceCenters,
                    InterfacePacketRotation) const;
    };
}

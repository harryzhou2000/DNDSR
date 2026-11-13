#pragma once

#include "Geom/Unstructured/UnstructuredMesh.hpp"
#include "DNDS/Runtime/Communication/CommunicationContext.hpp"

namespace DNDS::Cauchy
{
    /** @brief Detached numerical output with a retained immutable mesh read.
     * @details Owns prepared host cell fields and serialized surface rows. It
     * retains no solution/reconstruction/workspace or physics worker. Geometry
     * remains shared and read-pinned until writing ends; mesh mutation requires
     * the output service to drain first. The existing VTKHDF/surface schemas
     * and reference-frame coordinates are preserved. Write performs collective
     * file I/O only, on the supplied retained communicator.
     */
    class PreparedVisualization
    {
        friend class FiniteVolume;
        struct Data;
        std::shared_ptr<const Data> _data;
        PreparedVisualization(std::shared_ptr<Geom::UnstructuredMesh>, Runtime::Array<real>,
                              std::vector<std::string>, std::string surfaceCSV, real time, int deflate);

    public:
        void Write(const std::string &prefix, std::shared_ptr<Runtime::CommunicationContext>) const;
    };
}

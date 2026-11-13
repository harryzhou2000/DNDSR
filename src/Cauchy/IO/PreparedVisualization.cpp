#include "Cauchy/IO/PreparedVisualization.hpp"
#include "DNDS/MPI/CollectiveStage.hpp"
#include <fstream>

namespace DNDS::Cauchy
{
    struct PreparedVisualization::Data
    {
        std::shared_ptr<Geom::UnstructuredMesh> mesh;
        Geom::MeshReadAccess read;
        Runtime::Array<real> fields;
        std::vector<std::string> names;
        std::string surface;
        real time;
        Geom::UnstructuredMesh::HDF5OutSetting settings;
        Data(std::shared_ptr<Geom::UnstructuredMesh> geometry, Runtime::Array<real> values,
             std::vector<std::string> labels, std::string csv, real when, int deflate)
            : mesh(std::move(geometry)), read(mesh->AcquireRead(Geom::MeshReadRequirements::CellFlux())),
              fields(std::move(values)), names(std::move(labels)), surface(std::move(csv)), time(when),
              settings(mesh->hdf5OutSetting)
        {
            settings.deflateLevel = deflate;
            DNDS_check_throw_info(fields.GetDevice() == Runtime::Device::Host() &&
                                      fields.RowCount() == mesh->NumCell() && fields.Layout().Width() == index(names.size()),
                                  "visualization snapshot fields and mesh differ");
        }
    };
    PreparedVisualization::PreparedVisualization(std::shared_ptr<Geom::UnstructuredMesh> mesh, Runtime::Array<real> fields,
                                                 std::vector<std::string> names, std::string surfaceCSV, real time, int deflate)
        : _data(std::make_shared<Data>(std::move(mesh), std::move(fields), std::move(names), std::move(surfaceCSV), time, deflate)) {}
    void PreparedVisualization::Write(const std::string &prefix, std::shared_ptr<Runtime::CommunicationContext> communication) const
    {
        DNDS_check_throw_info(communication, "visualization writer needs a retained I/O communicator");
        const auto &data = *_data;
        const MPIInfo mpi(communication->Communicator());
        const auto read = data.fields.Read();
        const auto values = read.View<Runtime::TableRows>();
        data.mesh->PrintParallelVTKHDFDataArray(prefix, "", int(data.names.size()), 0, 0, 0, [&](int i)
                                                { return data.names.at(i); }, [&](int i, index iCell)
                                                { return values(iCell, i); }, {}, {}, {}, {}, {}, {}, data.time, mpi.comm, &data.settings);
        MPI::LocalStage(mpi, [&]
                        {
            std::ofstream file(prefix + "_surface_rank" + std::to_string(mpi.rank) + ".csv");
            file << data.surface;
            file.close();
            DNDS_check_throw_info(bool(file), "surface snapshot write failed"); });
    }
}

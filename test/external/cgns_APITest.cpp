#include <cgnslib.h>
#include <iostream>
#include <vector>

int main()
{
    int cgns_file;
    int cgerr;
    if (cg_open("test1.cgns", CG_MODE_WRITE, &cgns_file) != CG_OK)
        cg_error_exit();
    int cgns_base;
    if (cg_base_write(cgns_file, "my_good_cgns_base", 2, 3, &cgns_base) != CG_OK)
        cg_error_exit();
    int cgns_zone;
    cgsize_t zone_sizes[3] = {20, 12, 0};
    if (cg_zone_write(cgns_file, cgns_base, "my_good_cgns_zone", zone_sizes, Unstructured, &cgns_zone))
        cg_error_exit();
    std::vector<double> x(20), y(20), z(20);
    for (int i = 0; i < 5; i++)
        for (int j = 0; j < 4; j++)
        {
            x[i * 4 + j] = i;
            y[i * 4 + j] = j;
            z[i * 4 + j] = 0;
        }
    // std::cout << "good " << std::endl;
    int cgns_coord;
    double *xx = x.data();
    double **xxx = &xx;
    double ***xxxx = &xxx;
    if (cg_coord_write(cgns_file, cgns_base, cgns_zone, RealDouble, "CoordinateX", x.data(), &cgns_coord))
        cg_error_exit();
    // std::cout << "good2 " << std::endl;
    if (cg_coord_write(cgns_file, cgns_base, cgns_zone, RealDouble, "CoordinateY", y.data(), &cgns_coord))
        cg_error_exit();
    if (cg_coord_write(cgns_file, cgns_base, cgns_zone, RealDouble, "CoordinateZ", z.data(), &cgns_coord))
        cg_error_exit();
    int cgns_section;
    cgsize_t cell2nodes[12][4];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 3; j++)
        {
            cell2nodes[i * 3 + j][0] = i * 4 + j + 1;
            cell2nodes[i * 3 + j][1] = (i + 1) * 4 + j + 1;
            cell2nodes[i * 3 + j][2] = (i + 1) * 4 + j + 1 + 1;
            cell2nodes[i * 3 + j][3] = i * 4 + j + 1 + 1;
        }
    std::cout << "good2 " << std::endl;
    if (cg_section_write(cgns_file, cgns_base, cgns_zone, "Elem", QUAD_4, 0, 11, 0, cell2nodes[0], &cgns_section))
        cg_error_exit();
    if (cg_close(cgns_file))
        cg_error_exit();
    return 0;
}
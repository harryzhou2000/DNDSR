#include "json.hpp"
#include <iostream>

int main()
{
    using namespace nlohmann;

    json j = R"(
        {
            "main":
            {
                "test":
                [
                    1,2,3
                ]
            }
        }
    )"_json;

    std::cout << std::setw(4) << j << std::endl;
    std::cout << std::setw(4) << j["/main/test/0"_json_pointer] << std::endl;
    j["/main/test_2/test_2_1"_json_pointer] = json::object();
    j["/main/test_2//test_2_2"_json_pointer] = json::object();
    j["/main/test_2/../test_2_2"_json_pointer] = json::object();
    std::cout << std::setw(4) << j << std::endl;
    assert(j["/main/test_2/test_2_2"_json_pointer].is_object() == false);

    return 0;
}
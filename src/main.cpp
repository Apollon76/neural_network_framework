#include <iostream>
#include <fstream>
#include <cereal/types/vector.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/archives/json.hpp>

class BaseClass {
public:
    virtual void sayType() = 0;
};

// A class derived from BaseClass
class DerivedClassOne : public BaseClass {
public:
    DerivedClassOne() {}
    DerivedClassOne(int _x): x(_x) {}

    void sayType() override {
        std::cout << "first" << std::endl;
    }

    int x;

    template<class Archive>
    void serialize(Archive& ar) { ar(x); }
};

CEREAL_REGISTER_TYPE(DerivedClassOne)
CEREAL_REGISTER_POLYMORPHIC_RELATION(BaseClass, DerivedClassOne)


class EmbarrassingDerivedClass : public BaseClass {
public:
    EmbarrassingDerivedClass() {}

    EmbarrassingDerivedClass(float _y): y(_y) {}
    void sayType() override {
        std::cout << "second" << std::endl;
    }

    std::vector<float> y;

    template<class Archive>
    void serialize(Archive& ar) { ar(y); }
};

CEREAL_REGISTER_TYPE(EmbarrassingDerivedClass)
CEREAL_REGISTER_POLYMORPHIC_RELATION(BaseClass, EmbarrassingDerivedClass)

int main() {
    {
        std::ofstream os("polymorphism_test.json");
        cereal::JSONOutputArchive oarchive(os);

        // Create instances of the derived classes, but only keep base class pointers
        std::shared_ptr<BaseClass> ptr1 = std::make_shared<DerivedClassOne>();
        std::shared_ptr<BaseClass> ptr2 = std::make_shared<EmbarrassingDerivedClass>();
        oarchive(ptr1, ptr2);
    }

    {
        std::ifstream is("polymorphism_test.json");
        cereal::JSONInputArchive iarchive(is);

        // De-serialize the data as base class pointers, and watch as they are
        // re-instantiated as derived classes
        std::shared_ptr<BaseClass> ptr1;
        std::shared_ptr<BaseClass> ptr2;
        iarchive(ptr1, ptr2);

        // Ta-da! This should output:
        ptr1->sayType();  // "DerivedClassOne"
        ptr2->sayType();  // "EmbarrassingDerivedClass. Wait.. I mean DerivedClassTwo!"
        std::shared_ptr<DerivedClassOne> ptr_derived = std::dynamic_pointer_cast<DerivedClassOne>(ptr1);
        std::cout << ptr_derived->x << std::endl;
    }
}
#include <iostreams>
#include <cereal/types/polymorphic.hpp>
#include <cereal/archives/json.hpp>

struct BaseClass {
    virtual void sayType() = 0;
};

// A class derived from BaseClass
struct DerivedClassOne : public BaseClass {
    void sayType() override {
        std::cout << "first" << std::endl;
    }

    int x;

    template<class Archive>
    void serialize(Archive& ar) { ar(x); }
};

CEREAL_REGISTER_TYPE(DerivedClassOne);
CEREAL_REGISTER_POLYMORPHIC_RELATION(BaseClass, DerivedClassOne
)


struct EmbarrassingDerivedClass : public BaseClass {
    void sayType() override {
        std::cout << "second" << std::endl;
    }

    float y;

    template<class Archive>
    void serialize(Archive& ar) { ar(y); }
};

CEREAL_REGISTER_TYPE(EmbarrassingDerivedClass);
CEREAL_REGISTER_POLYMORPHIC_RELATION(BaseClass, EmbarrassingDerivedClass
)

int main() {
    {
        std::ofstream os("polymorphism_test.xml");
        cereal::J oarchive(os);

        // Create instances of the derived classes, but only keep base class pointers
        std::shared_ptr <BaseClass> ptr1 = std::make_shared<DerivedClassOne>();
        std::shared_ptr <BaseClass> ptr2 = std::make_shared<EmbarrassingDerivedClass>();
        oarchive(ptr1, ptr2);
    }

    {
        std::ifstream is("polymorphism_test.xml");
        cereal::XMLInputArchive iarchive(is);

        // De-serialize the data as base class pointers, and watch as they are
        // re-instantiated as derived classes
        std::shared_ptr <BaseClass> ptr1;
        std::shared_ptr <BaseClass> ptr2;
        iarchive(ptr1, ptr2);

        // Ta-da! This should output:
        ptr1->sayType();  // "DerivedClassOne"
        ptr2->sayType();  // "EmbarrassingDerivedClass. Wait.. I mean DerivedClassTwo!"
    }

    return 0;
}
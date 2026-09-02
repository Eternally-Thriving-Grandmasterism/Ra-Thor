// RaThor/Geometry/MercyThresholdLeanCpp.cpp
// Full Working Lean 4 → C++ FFI Example with Error Handling
// Compile: g++ -o mercy_check MercyThresholdLeanCpp.cpp -I/path/to/lean/include -L/path/to/lean/lib -llean

#include <iostream>
#include <lean/lean.h>
#include <stdexcept>

// Lean-exported function (from Lean side)
extern "C" {
    bool lean_mercy_threshold_safe(double score, double valence);
}

int main() {
    try {
        double score = 0.96;
        double valence = 1.0;

        // Initialize Lean runtime with error checking
        if (!lean_initialize_runtime_module()) {
            throw std::runtime_error("Failed to initialize Lean runtime");
        }

        // Call the verified Lean 4 mercy threshold
        bool result = lean_mercy_threshold_safe(score, valence);

        if (result) {
            std::cout << "Mercy threshold PASSED (score=" << score 
                      << ", valence=" << valence << ")" << std::endl;
        } else {
            std::cout << "Mercy threshold FAILED" << std::endl;
            return 1;  // Non-zero exit on failure
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

// Lean side (IntervalMercy.lean):
// @[export] def mercy_threshold_safe (score : Float) (valence : Float) : Bool :=
//   score > 0.95 && valence >= 0.999999

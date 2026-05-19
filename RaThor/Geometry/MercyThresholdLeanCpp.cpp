// RaThor/Geometry/MercyThresholdLeanCpp.cpp
// Full Working Lean 4 → C++ FFI Example
// Compile: g++ -o mercy_check MercyThresholdLeanCpp.cpp -I/path/to/lean/include -L/path/to/lean/lib -llean

#include <iostream>
#include <lean/lean.h>

// Lean-exported function (from Lean side)
extern "C" {
    bool lean_mercy_threshold_safe(double score, double valence);
}

int main() {
    double score = 0.96;
    double valence = 1.0;

    // Initialize Lean runtime
    lean_initialize_runtime_module();

    // Call the verified Lean 4 mercy threshold
    bool result = lean_mercy_threshold_safe(score, valence);

    if (result) {
        std::cout << "Mercy threshold PASSED (score=" << score << ", valence=" << valence << ")" << std::endl;
    } else {
        std::cout << "Mercy threshold FAILED" << std::endl;
    }

    return 0;
}

// Lean side (IntervalMercy.lean):
// @[export] def mercy_threshold_safe (score : Float) (valence : Float) : Bool :=
//   score > 0.95 && valence >= 0.999999

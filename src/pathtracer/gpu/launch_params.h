#pragma once

#include "optix.h"

namespace osc {

struct LaunchParams {
    int frameID{0};

    OptixTraversableHandle traversable;
};

} // namespace osc
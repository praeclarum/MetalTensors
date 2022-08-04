using System;
using MetalPerformanceShaders;

namespace MetalTensors
{
    public enum RegularizationType
    {
        None = (int)MPSNNRegularizationType.None,
        L1 = (int)MPSNNRegularizationType.L1,
        L2 = (int)MPSNNRegularizationType.L2,
    }
}

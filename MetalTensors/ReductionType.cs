using System;
using MetalPerformanceShaders;

namespace MetalTensors
{
    public enum ReductionType
    {
        None = (int)MPSCnnReductionType.None,
        Mean = (int)MPSCnnReductionType.Mean,
        Sum = (int)MPSCnnReductionType.Sum,
    }
}

using System;
using MetalPerformanceShaders;

namespace MetalTensors
{
    public enum LossType
    {
        CategoricalCrossEntropy = (int)MPSCnnLossType.CategoricalCrossEntropy,
        Hinge = (int)MPSCnnLossType.Hinge,
        MeanAbsoluteError = (int)MPSCnnLossType.MeanAbsoluteError,
        MeanSquaredError = (int)MPSCnnLossType.MeanSquaredError,
        //SigmoidCrossEntropy = (int)MPSCnnLossType.SigmoidCrossEntropy,
        //SoftMaxCrossEntropy = (int)MPSCnnLossType.SoftMaxCrossEntropy,
    }
}

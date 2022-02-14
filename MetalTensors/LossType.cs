using System;
using MetalPerformanceShaders;

namespace MetalTensors
{
    public enum LossType
    {
        /// <summary>
        /// losses = |y - t|
        /// </summary>
        MeanAbsoluteError = (int)MPSCnnLossType.MeanAbsoluteError,

        /// <summary>
        /// losses = (y - t)^2
        /// </summary>
        MeanSquaredError = (int)MPSCnnLossType.MeanSquaredError,

        /// <summary>
        /// losses = -t * LogSoftMax(y)
        /// </summary>
        SoftMaxCrossEntropy = (int)MPSCnnLossType.SoftMaxCrossEntropy,

        /// <summary>
        /// losses = max(y, 0) - y * t + log(1 + exp(-|y|))
        /// </summary>
        SigmoidCrossEntropy = (int)MPSCnnLossType.SigmoidCrossEntropy,

        /// <summary>
        /// losses = -t * log(y)
        /// </summary>
        CategoricalCrossEntropy = (int)MPSCnnLossType.CategoricalCrossEntropy,

        /// <summary>
        /// losses = max(1 - (t * y), 0.0f)
        /// </summary>
        Hinge = (int)MPSCnnLossType.Hinge,

        /// <summary>
        /// losses = -(t * log(y + epsilon)) - ((1 - t) * log(1 - y + epsilon))
        /// </summary>
        Log = (int)MPSCnnLossType.Log,

        /// <summary>
        /// losses = t * (log(t) - y)
        /// </summary>
        KLDivergence = (int)MPSCnnLossType.KullbackLeiblerDivergence,
    }
}

using System;

namespace MetalTensors
{
    /// <summary>
    /// This is a labelled (output) loss. Do not inherit from this class.
    /// </summary>
    public abstract class Loss
    {
        public static readonly Loss CategoricalCrossEntropy = new BuiltinLoss(LossType.CategoricalCrossEntropy);
        public static readonly Loss Hinge = new BuiltinLoss (LossType.Hinge);
        public static readonly Loss MeanAbsoluteError = new BuiltinLoss (LossType.MeanAbsoluteError);
        public static readonly Loss MeanSquaredError = new BuiltinLoss (LossType.MeanSquaredError);
        public static readonly Loss SigmoidCrossEntropy = new BuiltinLoss (LossType.SigmoidCrossEntropy);
        public static readonly Loss SoftMaxCrossEntropy = new BuiltinLoss (LossType.SoftMaxCrossEntropy);

        public static Loss Custom (Func<Tensor, Tensor, Tensor> lossFunction) => new CustomLoss (lossFunction);

        protected Loss ()
        {
        }
    }

    public class BuiltinLoss : Loss
    {
        public LossType LossType { get; }

        public BuiltinLoss (LossType lossType)
        {
            LossType = lossType;
        }

        public override string ToString () => $"Loss.{LossType}";
    }

    public class CustomLoss : Loss
    {
        public Func<Tensor, Tensor, Tensor> LossFunction { get; }

        public CustomLoss (Func<Tensor, Tensor, Tensor> lossFunction)
        {
            LossFunction = lossFunction;
        }
    }
}

using System;

namespace MetalTensors
{
    public abstract class Loss
    {
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

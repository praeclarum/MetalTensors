using System;

namespace MetalTensors
{
    /// <summary>
    /// This is a labelled (output) loss. Do not inherit from this class.
    /// </summary>
    public abstract class Loss
    {
        /// <summary>
        /// loss = mean(|y - t|)
        /// </summary>
        public static readonly Loss MeanAbsoluteError = Builtin (LossType.MeanAbsoluteError);

        /// <summary>
        /// loss = mean((y - t)^2)
        /// </summary>
        public static readonly Loss MeanSquaredError = Builtin (LossType.MeanSquaredError);

        /// <summary>
        /// loss = mean(-t * LogSoftMax(y))
        /// </summary>
        public static readonly Loss SoftMaxCrossEntropy = Builtin (LossType.SoftMaxCrossEntropy);
        /// <summary>
        /// loss = sum(-t * LogSoftMax(y))
        /// </summary>
        public static readonly Loss SumSoftMaxCrossEntropy = Builtin (LossType.SoftMaxCrossEntropy, ReductionType.Sum);

        /// <summary>
        /// loss = mean(max(y, 0) - y * t + log(1 + exp(-|y|)))
        /// </summary>
        public static readonly Loss SigmoidCrossEntropy = Builtin (LossType.SigmoidCrossEntropy);
        /// <summary>
        /// loss = sum(max(y, 0) - y * t + log(1 + exp(-|y|)))
        /// </summary>
        public static readonly Loss SumSigmoidCrossEntropy = Builtin (LossType.SigmoidCrossEntropy, ReductionType.Sum);

        /// <summary>
        /// loss = mean(-t * log(y))
        /// </summary>
        public static readonly Loss CategoricalCrossEntropy = new BuiltinLoss(LossType.CategoricalCrossEntropy);
        /// <summary>
        /// loss = sum(-t * log(y))
        /// </summary>
        public static readonly Loss SumCategoricalCrossEntropy = new BuiltinLoss(LossType.CategoricalCrossEntropy, ReductionType.Sum);

        /// <summary>
        /// loss = mean(max(1 - (t * y), 0.0f))
        /// </summary>
        public static readonly Loss Hinge = Builtin (LossType.Hinge);
        /// <summary>
        /// loss = sum(max(1 - (t * y), 0.0f))
        /// </summary>
        public static readonly Loss SumHinge = Builtin (LossType.Hinge, ReductionType.Sum);

        /// <summary>
        /// loss = mean(-(t * log(y + epsilon)) - ((1 - t) * log(1 - y + epsilon)))
        /// </summary>
        public static readonly Loss Log = Builtin (LossType.Log);
        /// <summary>
        /// loss = sum(-(t * log(y + epsilon)) - ((1 - t) * log(1 - y + epsilon)))
        /// </summary>
        public static readonly Loss SumLog = Builtin (LossType.Log, ReductionType.Sum);

        /// <summary>
        /// loss = mean(t * (log(t) - y))
        /// </summary>
        public static readonly Loss KLDivergence = Builtin (LossType.KLDivergence);
        /// <summary>
        /// loss = sum(t * (log(t) - y))
        /// </summary>
        public static readonly Loss SumKLDivergence = Builtin (LossType.KLDivergence, ReductionType.Sum);

        public static Loss Custom (Func<Tensor, Tensor, Tensor> lossFunction) => new CustomLoss (lossFunction);
        public static Loss Builtin (LossType lossType, ReductionType reductionType = ReductionType.Mean) => new BuiltinLoss (lossType, reductionType);

        protected Loss ()
        {
        }

        public abstract Tensor Call (Tensor y, Tensor t, float weight);
    }

    public class BuiltinLoss : Loss
    {
        public LossType LossType { get; }
        public ReductionType ReductionType { get; }

        public BuiltinLoss (LossType lossType, ReductionType reductionType = ReductionType.Mean)
        {
            LossType = lossType;
            ReductionType = reductionType;
        }

        public override string ToString () => $"{LossType} Loss (Reduction={ReductionType})";

        public override Tensor Call (Tensor prediction, Tensor truth, float weight)
        {
            //var i = prediction;
            //var lossType = Model.DefaultLossType;
            //if (prediction is LayerTensor lt) {
            //    if (lt.Layer is SigmoidLayer) {
            //        i = lt.Inputs[0];
            //        lossType = LossType.SigmoidCrossEntropy;
            //    }
            //    else if (lt.Layer is SoftMaxLayer) {
            //        i = lt.Inputs[0];
            //        lossType = LossType.SoftMaxCrossEntropy;
            //    }
            //}

            var layer = new Layers.LossLayer (prediction.Label + " Loss", LossType, ReductionType, weight);
            return layer.GetOutput (prediction, truth);
        }
    }

    public class CustomLoss : Loss
    {
        public Func<Tensor, Tensor, Tensor> LossFunction { get; }

        public CustomLoss (Func<Tensor, Tensor, Tensor> lossFunction)
        {
            LossFunction = lossFunction;
        }

        public override Tensor Call (Tensor y, Tensor t, float weight)
        {
            var loss = LossFunction (y, t);
            if (Math.Abs (weight - 1.0) > 1e-9)
                loss *= weight;
            return loss;
        }
    }
}

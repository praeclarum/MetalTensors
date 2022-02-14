using System;

namespace MetalTensors
{
    /// <summary>
    /// This is a labelled (output) loss. Do not inherit from this class.
    /// </summary>
    public abstract class Loss
    {
        public static readonly Loss CategoricalCrossEntropy = new BuiltinLoss(LossType.CategoricalCrossEntropy);
        public static readonly Loss SumCategoricalCrossEntropy = new BuiltinLoss(LossType.CategoricalCrossEntropy, ReductionType.Sum);
        public static readonly Loss Hinge = new BuiltinLoss (LossType.Hinge);
        public static readonly Loss MeanAbsoluteError = new BuiltinLoss (LossType.MeanAbsoluteError);
        public static readonly Loss MeanSquaredError = new BuiltinLoss (LossType.MeanSquaredError);
        public static readonly Loss SigmoidCrossEntropy = new BuiltinLoss (LossType.SigmoidCrossEntropy);
        public static readonly Loss SumSigmoidCrossEntropy = new BuiltinLoss (LossType.SigmoidCrossEntropy, ReductionType.Sum);
        public static readonly Loss SoftMaxCrossEntropy = new BuiltinLoss (LossType.SoftMaxCrossEntropy);
        public static readonly Loss SumSoftMaxCrossEntropy = new BuiltinLoss (LossType.SoftMaxCrossEntropy, ReductionType.Sum);

        public static Loss Custom (Func<Tensor, Tensor, Tensor> lossFunction) => new CustomLoss (lossFunction);

        protected Loss ()
        {
        }

        public abstract Tensor Call (Tensor prediction, Tensor truth, Tensor? weights = null);
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

        public override Tensor Call (Tensor prediction, Tensor truth, Tensor? weights = null)
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

            var layer = new Layers.LossLayer (prediction.Label + " Loss", LossType, ReductionType);
            return weights != null ?
                layer.GetOutput (prediction, truth, weights) :
                layer.GetOutput (prediction, truth);
        }
    }

    public class CustomLoss : Loss
    {
        public Func<Tensor, Tensor, Tensor> LossFunction { get; }

        public CustomLoss (Func<Tensor, Tensor, Tensor> lossFunction)
        {
            LossFunction = lossFunction;
        }

        public override Tensor Call (Tensor prediction, Tensor truth, Tensor? weights = null)
        {
            return LossFunction (prediction, truth);
        }
    }
}

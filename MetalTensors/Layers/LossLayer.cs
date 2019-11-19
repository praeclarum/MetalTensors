using System;
using System.Linq;
using Foundation;
using Metal;
using MetalPerformanceShaders;

namespace MetalTensors.Layers
{
    public class LossLayer : Layer
    {
        static readonly int[] scalarShape = { 1 };

        public override int MinInputCount => 2;

        public LossType LossType { get; }
        public ReductionType ReductionType { get; }

        public LossLayer (string? label, LossType lossType, ReductionType reductionType)
            : base (label)
        {
            LossType = lossType;
            ReductionType = reductionType;
        }

        public override void ValidateInputShapes (params Tensor[] inputs)
        {
            base.ValidateInputShapes (inputs);
            if (inputs.Length != 2) {
                throw new ArgumentException ("Loss requires two inputs: data and labels", nameof (inputs));
            }
            var inputShape = inputs[0].Shape;
            var labelsShape = inputs[1].Shape;
            if (!inputShape.ShapeEquals (labelsShape)) {
                throw new ArgumentException ($"Labels shape {labelsShape.ToShapeString ()} must match the data shape {inputShape.ToShapeString ()}", nameof (inputs));
            }
        }

        public override int[] GetOutputShape (params Tensor[] inputs)
        {
            return ReductionType == ReductionType.None ? inputs[0].Shape : scalarShape;
        }

        protected override MPSNNFilterNode CreateFilterNode ((MPSNNImageNode ImageNode, int[] Shape)[] inputs, IMTLDevice device)
        {
            var sourceNodes = inputs.Select (x => x.ImageNode).ToArray ();
            var descriptor = MPSCnnLossDescriptor.Create ((MPSCnnLossType)LossType, (MPSCnnReductionType)ReductionType);
            var ln = new MPSNNForwardLossNode (sourceNodes, descriptor);
            return ln;
        }
    }
}

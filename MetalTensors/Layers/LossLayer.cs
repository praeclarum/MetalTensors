using System;
using System.Linq;
using Foundation;
using Metal;
using MetalPerformanceShaders;

namespace MetalTensors.Layers
{
    public class LossLayer : Layer
    {
        public override int InputCount => 2;

        public LossType LossType { get; }
        public MPSCnnReductionType ReductionType { get; }

        public LossLayer (string? label, LossType lossType, MPSCnnReductionType reductionType)
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
                throw new ArgumentException ("Labels shape must match the data shape", nameof (inputs));
            }
        }

        public override int[] GetOutputShape (params Tensor[] inputs)
        {
            return inputs[0].Shape;
        }

        protected override MPSNNFilterNode CreateFilterNode ((MPSNNImageNode ImageNode, int[] Shape)[] inputs, IMTLDevice device)
        {
            var sourceNodes = inputs.Select (x => x.ImageNode).ToArray ();
            var descriptor = MPSCnnLossDescriptor.Create ((MPSCnnLossType)LossType, ReductionType);
            var ln = new MPSNNForwardLossNode (sourceNodes, descriptor);
            var resultImage = ln.ResultImage;
            resultImage.ExportFromGraph = true;
            resultImage.SynchronizeResource = true;
            resultImage.ImageAllocator = MPSImage.DefaultAllocator;
            return ln;
        }
    }
}

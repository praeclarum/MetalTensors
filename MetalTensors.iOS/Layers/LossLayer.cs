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

        public LossLayer (LossType lossType, MPSCnnReductionType reductionType)
        {
            LossType = lossType;
            ReductionType = reductionType;
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

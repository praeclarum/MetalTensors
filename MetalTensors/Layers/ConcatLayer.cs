using System;
using System.Linq;
using Metal;
using MetalPerformanceShaders;

namespace MetalTensors.Layers
{
    public class ConcatLayer : Layer
    {
        public override int MinInputCount => 1;

        public override int[] GetOutputShape (params Tensor[] inputs)
        {
            var inputShape = inputs[0].Shape;

            var outputShape = new int[inputShape.Length];
            var nc = 0;
            foreach (var i in inputs) {
                nc += ((i.Shape[^1] + 3) / 4) * 4;
            }
            for (var i = 0; i < inputShape.Length; i++) {
                var s = inputShape[i];
                outputShape[i] = s;
            }
            outputShape[^1] = nc;
            return outputShape;
        }

        protected override MPSNNFilterNode CreateFilterNode ((MPSNNImageNode ImageNode, int[] Shape)[] inputs, IMTLDevice device)
        {
            return new MPSNNConcatenationNode (inputs.Select (x => x.ImageNode).ToArray ());
        }
    }
}

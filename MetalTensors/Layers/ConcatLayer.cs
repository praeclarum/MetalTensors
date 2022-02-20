using System;
using System.Linq;
using Metal;
using MetalPerformanceShaders;

namespace MetalTensors.Layers
{
    public class ConcatLayer : Layer
    {
        public override int MinInputCount => 1;

        public ConcatLayer (string? name = null)
            : base (name)
        {
        }

        public override void ValidateInputShapes (params Tensor[] inputs)
        {
            base.ValidateInputShapes (inputs);

            var shape = inputs[0].Shape;
            var nc = shape.Length - 1;
            if (nc == 0)
                return;

            foreach (var i in inputs.Skip (1)) {
                var s = i.Shape;
                if (s.Length != shape.Length) {
                    throw new ArgumentException ($"Mismatched input shape dimensions in concat", nameof (inputs));
                }
                for (var j = 0; j < nc; j++) {
                    if (s[j] != shape[j]) {
                        throw new ArgumentException ($"Mismatched input shapes in concat. Expected {shape.ToShapeString ()}, got {s.ToShapeString ()}.", nameof (inputs));
                    }
                }
            }
        }

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

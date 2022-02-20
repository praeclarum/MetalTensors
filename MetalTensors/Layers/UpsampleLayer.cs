using System;
using Metal;
using MetalPerformanceShaders;

namespace MetalTensors.Layers
{
    public class UpsampleLayer : Layer
    {
        public int ScaleX { get; }
        public int ScaleY { get; }

        public override int MinInputCount => 1;

        public UpsampleLayer (int scaleX, int scaleY, string? name = null)
            : base (name)
        {
            if (scaleX < 1)
                throw new ArgumentException ("Scale must be >= 1 for upsampling", nameof (scaleX));
            if (scaleY < 1)
                throw new ArgumentException ("Scale must be >= 1 for upsampling", nameof (scaleY));
            ScaleX = scaleX;
            ScaleY = scaleY;
        }

        public override Config Config => base.Config.Update (new Config {
            { "scaleX", ScaleX },
            { "scaleY", ScaleY },
        });

        public override void ValidateInputShapes (params Tensor[] inputs)
        {
            base.ValidateInputShapes (inputs);

            var inputShape = inputs[0].Shape;
            if (inputShape.Length < 3)
                throw new ArgumentException ($"Upsample inputs must have 3 dimensions HxWxC ({inputs.Length} given)", nameof (inputs));
        }

        public override int[] GetOutputShape (params Tensor[] inputs)
        {
            var inShape = inputs[0].Shape;
            var h = inShape[0] * ScaleY;
            var w = inShape[1] * ScaleX;
            var c = inShape[2];
            var outShape = new[] { h, w, c };
            return outShape;
        }

        protected override MPSNNFilterNode CreateFilterNode ((MPSNNImageNode ImageNode, int[] Shape)[] inputs, IMTLDevice device)
        {
            return new MPSCnnUpsamplingNearestNode (inputs[0].ImageNode, (nuint)ScaleX, (nuint)ScaleY);
        }
    }
}

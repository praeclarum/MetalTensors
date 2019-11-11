using System;

namespace MetalTensors.Tensors
{
    public class ZeroTensor : Tensor
    {
        readonly int[] shape;

        public override int[] Shape => shape;

        public ZeroTensor (params int[] shape)
        {
            ValidateShape (shape);
            this.shape = shape;
        }

        public override void Copy (Span<float> destination)
        {
            var n = Math.Min (GetShapeLength (shape), destination.Length);
            for (var i = 0; i < n; i++) {
                destination[i] = 0.0f;
            }
        }
    }
}

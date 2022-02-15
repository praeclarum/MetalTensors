using System;
using System.Collections.Concurrent;
using Metal;
using MetalPerformanceShaders;

namespace MetalTensors.Tensors
{
    public abstract class PlaceholderTensor : Tensor
    {
        readonly int[] shape;

        public override int[] Shape => shape;

        public override bool IsStatic => false;

        protected PlaceholderTensor (string label, int[] shape)
            : base (label)
        {
            this.shape = shape.NormalizeShape ();
        }        

        public override void Copy (Span<float> destination, IMTLDevice device)
        {
            var n = ValidateCopyDestination (destination);
            for (var i = 0; i < n; i++) {
                destination[i] = 0.0f;
            }
        }

        public override MPSImage GetMetalImage (IMTLDevice device)
        {
            var image = MetalExtensions.CreateConstantImage (Shape, 0.0f);
            return image;
        }
    }
}

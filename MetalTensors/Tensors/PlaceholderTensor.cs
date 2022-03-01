using System;
using System.Collections.Concurrent;
using System.Threading.Tasks;
using Metal;
using MetalPerformanceShaders;

namespace MetalTensors.Tensors
{
    public abstract class PlaceholderTensor : Tensor
    {
        readonly int[] shape;

        public override int[] Shape => shape;

        public override bool IsStatic => false;

        protected PlaceholderTensor (string? name, int[] shape)
            : base (name)
        {
            this.shape = shape.NormalizeShape ();
        }

        public override Config Config => base.Config.Add ("shape", Shape);

        public override void CopyTo (Span<float> destination, IMTLDevice? device = null)
        {
            var n = ValidateCopyDestination (destination);
            for (var i = 0; i < n; i++) {
                destination[i] = 0.0f;
            }
        }

        public override Task CopyToAsync (MPSImage image, IMTLCommandQueue queue)
        {
            return Task.Run (() => {
                image.Fill (0.0f);
            });
        }

        public override void EncodeToCommandBuffer (MPSImage image, MPSCommandBuffer commands)
        {
            image.Fill (0.0f);
        }

        public override MPSImage GetMetalImage (IMTLDevice device)
        {
            var image = MetalHelpers.CreateConstantImage (Shape, 0.0f);
            return image;
        }
    }
}

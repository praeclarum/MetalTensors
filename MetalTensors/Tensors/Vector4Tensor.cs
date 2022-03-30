using System;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Numerics;
using System.Threading;
using System.Threading.Tasks;
using Foundation;
using Metal;
using MetalPerformanceShaders;

namespace MetalTensors.Tensors
{
    public class Vector4Tensor : Tensor
    {
        readonly int[] shape;

        public override int[] Shape => shape;

        public override bool IsStatic => true;

        public Vector4 Value { get; }

        public Vector4Tensor (Vector4 value, string? name)
            : base (name)
        {
            Value = value;
            this.shape = Shapes.GetShape (4);
        }

        public override Config Config => base.Config.Update (new Config {
            { "value", Value },
        });

        public override string ToString ()
        {
            return base.ToString () + "=" + Value;
        }

        public override void CopyTo (Span<float> destination, IMTLDevice? device = null)
        {
            ValidateCopyDestination (destination);
            var c = Value;
            destination[0] = c.X;
            destination[1] = c.Y;
            destination[2] = c.Z;
            destination[3] = c.W;
        }

        //public override Task CopyToAsync (MPSImage image, IMTLCommandQueue queue)
        //{
        //    return Task.Run (() => {
        //        image.Fill (ConstantValue);
        //    });
        //}

        //public override void EncodeToCommandBuffer (MPSImage image, MPSCommandBuffer commands)
        //{
        //    image.Fill (ConstantValue);
        //}

        //public override MPSImage GetMetalImage (IMTLDevice device)
        //{
        //    var image = MetalHelpers.CreateConstantImage (Shape, ConstantValue);
        //    return image;
        //}

    }
}

using System;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Threading;
using Foundation;
using Metal;
using MetalPerformanceShaders;

namespace MetalTensors.Tensors
{
    public class ConstantTensor : Tensor
    {
        readonly int[] shape;

        public override int[] Shape => shape;

        public float ConstantValue { get; }

        public ConstantTensor (string? label, float constant, params int[] shape)
            : base (label)
        {
            ConstantValue = constant;
            this.shape = shape.NormalizeShape ();
        }

        public ConstantTensor (float constant, params int[] shape)
            : this (null, constant, shape)
        {
        }

        public override void Copy (Span<float> destination, IMTLDevice device)
        {
            var n = ValidateCopyDestination (destination);
            var c = ConstantValue;
            for (var i = 0; i < n; i++) {
                destination[i] = c;
            }
        }

        public override MPSImage GetMetalImage (IMTLDevice device)
        {
            var image = CreateConstantImage (Shape, ConstantValue);
            return image;
        }
    }
}

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

        public override bool IsStatic => true;

        public float ConstantValue { get; }

        public ConstantTensor (float constant, int[] shape, string? name)
            : base (name)
        {
            ConstantValue = constant;
            this.shape = shape.NormalizeShape ();
        }

        public ConstantTensor (float constant, params int[] shape)
            : this (constant, shape, null)
        {
        }

        public ConstantTensor (float constant, Tensor mimic)
            : this (constant, mimic.Shape, null)
        {
        }

        protected override TensorHandle CreateHandle (string? label) => new ConstantHandle (this, label, ConstantValue);

        public override string ToString ()
        {
            return base.ToString () + "=" + ConstantValue;
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
            var image = MetalExtensions.CreateConstantImage (Shape, ConstantValue);
            return image;
        }

        public override Tensor Add (Tensor other)
        {
            if (other is ConstantTensor o)
                return new ConstantTensor (ConstantValue + o.ConstantValue, Shape);
            return other.Linear (1.0f, ConstantValue);
        }
        public override Tensor Add (float other) => new ConstantTensor (ConstantValue + other, Shape);
        public override Tensor Add (int other) => new ConstantTensor (ConstantValue + other, Shape);

        public override Tensor Divide (Tensor other)
        {
            if (other is ConstantTensor o)
                return new ConstantTensor (ConstantValue / o.ConstantValue, Shape);
            return base.Divide (other);
        }
        public override Tensor Divide (float other) => new ConstantTensor (ConstantValue / other, Shape);
        public override Tensor Divide (int other) => new ConstantTensor (ConstantValue / other, Shape);

        public override Tensor Multiply (Tensor other)
        {
            if (other is ConstantTensor o)
                return new ConstantTensor (ConstantValue * o.ConstantValue, Shape);
            return other.Linear (ConstantValue);
        }
        public override Tensor Multiply (float other) => new ConstantTensor (ConstantValue * other, Shape);
        public override Tensor Multiply (int other) => new ConstantTensor (ConstantValue * other, Shape);

        public override Tensor Subtract (Tensor other)
        {
            if (other is ConstantTensor o)
                return new ConstantTensor (ConstantValue - o.ConstantValue, Shape);
            return other.Linear(-1.0f, ConstantValue);
        }
        public override Tensor Subtract (float other) => new ConstantTensor (ConstantValue - other, Shape);
        public override Tensor Subtract (int other) => new ConstantTensor (ConstantValue - other, Shape);
    }
}

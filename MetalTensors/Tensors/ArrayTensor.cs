using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using Metal;
using MetalPerformanceShaders;

namespace MetalTensors.Tensors
{
    public class ArrayTensor : Tensor
    {
        readonly float[] data;
        readonly int[] shape;

        public override int[] Shape => shape;

        public override bool IsStatic => true;

        public ArrayTensor (int[] shape, float[] data)
        {
            this.data = data;
            this.shape = shape;
        }

        public ArrayTensor (float[] data)
        {
            this.data = data;
            this.shape = new int[] { data.Length };
        }

        public override void Copy (Span<float> destination, IMTLDevice device)
        {
            ValidateCopyDestination (destination);
            Span<float> dataSpan = data;
            dataSpan.CopyTo (destination);
        }

        public override float this[params int[] indexes] {
            get {
                var i = 0;
                var n = Math.Min (shape.Length, indexes.Length);
                var maxIndex = 1;
                for (var j = 0; j < n; j++) {
                    maxIndex *= shape[j];
                }
                for (var j = 0; j < n; j++) {
                    maxIndex /= shape[j];
                    i += indexes[j] * maxIndex;
                }
                return data[i];
            }
        }

        public override unsafe MPSImage GetMetalImage (IMTLDevice device)
        {
            var image = MetalHelpers.CreateUninitializedImage (Shape);
            fixed (float* dataPtr = data) {
                image.WriteBytes ((IntPtr)dataPtr, MPSDataLayout.HeightPerWidthPerFeatureChannels, 0);
            }
            return image;
        }
    }
}

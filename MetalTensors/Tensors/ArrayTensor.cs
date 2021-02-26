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

        public override void Copy (Span<float> destination)
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
            var image = CreateUninitializedImage (Shape);
            fixed (float* dataPtr = data) {
                image.WriteBytes ((IntPtr)dataPtr, MPSDataLayout.HeightPerWidthPerFeatureChannels, 0);
            }

#if DEBUG_ARRAY_TENSOR
            var dt = new MPSImageTensor (image);
            for (var i = 0; i < Math.Min (5, Shape[0]); i++) {
                var x = dt[i];
                Debug.Assert (Math.Abs (x - data[i]) < 1.0e-6f);
            }
#endif

            return image;
        }
    }
}

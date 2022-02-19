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

        [ConfigCtor]
        public ArrayTensor (int[] shape)
        {
            this.shape = shape;
            var len = 1;
            foreach (var s in shape) {
                len *= s;
            }
            data = new float[len];
        }

        public ArrayTensor (int[] shape, float[] data)
        {
            this.data = data;
            this.shape = shape;
        }

        public ArrayTensor (float[] data)
        {
            this.data = data;
            shape = new int[] { data.Length };
        }

        public override Config Config => base.Config.Update (new Config {
            { "shape", Shape },
        });

        public override void Copy (Span<float> destination, IMTLDevice? device = null)
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

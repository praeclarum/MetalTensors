using System;
using System.Collections.Concurrent;
using Metal;
using MetalPerformanceShaders;

namespace MetalTensors.Tensors
{
    public class ArrayTensor : Tensor
    {
        readonly float[] data;
        readonly int[] shape;

        public override int[] Shape => shape;

        readonly ConcurrentDictionary<IntPtr, MPSImage> deviceImages =
            new ConcurrentDictionary<IntPtr, MPSImage> ();

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
                if (indexes.Length > 0)
                    i = indexes[0];
                return data[i];
            }
        }

        public override unsafe MPSImage GetMetalImage (IMTLDevice device)
        {
            var key = device.Handle;
            if (deviceImages.TryGetValue (key, out var image))
                return image;

            image = CreateConstantImage (Shape, 0.0f);
            fixed (float* dataPtr = data) {
                image.WriteBytes ((IntPtr)dataPtr, MPSDataLayout.HeightPerWidthPerFeatureChannels, 0);
            }

            if (deviceImages.TryAdd (key, image))
                return image;
            return deviceImages[key];
        }
    }
}

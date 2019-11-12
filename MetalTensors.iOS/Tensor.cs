using System;
using Foundation;
using Metal;
using MetalTensors.Tensors;

namespace MetalTensors
{
    public abstract class Tensor
    {
        public abstract int[] Shape { get; }

        public virtual float Item {
            get {
                Span<float> span = stackalloc float[1];
                Copy (span);
                return span[0];
            }
        }

        public abstract void Copy (Span<float> destination);

        public static void ValidateShape (params int[] shape)
        {
            for (var i = 0; i < shape.Length; i++) {
                if (shape[i] <= 0)
                    throw new ArgumentOutOfRangeException (nameof (shape), $"Shape dimension must be > 0");
            }
        }

        protected void ValidateCopyDestination (Span<float> destination)
        {
            var neededLength = GetShapeLength (Shape);
            if (neededLength > destination.Length) {
                throw new ArgumentOutOfRangeException (nameof (destination), "Tensor copy destination memory is too small");
            }
        }

        public static int GetShapeLength (params int[] shape)
        {
            var r = 1;
            for (var i = 0; i < shape.Length; i++) {
                r *= shape[i];
            }
            return r;
        }

        public static Tensor ReadImage (NSUrl url, int featureChannels = 3, IMTLDevice? device = null)
        {
            return new MPSImageTensor (url, featureChannels, device);
        }

        public static Tensor ReadImage (string path, int featureChannels = 3, IMTLDevice? device = null)
        {
            return new MPSImageTensor (path, featureChannels, device);
        }
    }
}

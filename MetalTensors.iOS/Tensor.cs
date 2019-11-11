using System;

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
                    throw new ArgumentOutOfRangeException (nameof(shape), $"Shape dimension must be > 0");
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
    }
}

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace MetalTensors
{
    public abstract class Tensor
    {
        public abstract int[] Shape { get; }

        public void ValidateShape (int[] shape, string argumentName = "shape")
        {
            for (var i = 0; i < shape.Length; i++) {
                if (shape[i] <= 0)
                    throw new ArgumentOutOfRangeException (argumentName, $"Shape dimension must be > 0");
            }
        }
    }
}

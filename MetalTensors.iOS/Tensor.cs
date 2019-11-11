using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace MetalTensors
{
    public abstract class Tensor
    {
        public abstract int[] Shape { get; }
    }
}

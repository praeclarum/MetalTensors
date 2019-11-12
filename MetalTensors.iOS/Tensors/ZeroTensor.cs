using System;

namespace MetalTensors.Tensors
{
    public class ZeroTensor : ConstantTensor
    {
        public ZeroTensor (params int[] shape)
            : base (0.0f, shape)
        {
        }
    }
}

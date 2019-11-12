using System;

namespace MetalTensors.Tensors
{
    public class OnesTensor : ConstantTensor
    {
        public OnesTensor (params int[] shape)
            : base (1.0f, shape)
        {
        }
    }
}

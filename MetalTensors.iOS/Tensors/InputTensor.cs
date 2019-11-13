using System;

namespace MetalTensors.Tensors
{
    public class InputTensor : PlaceholderTensor
    {
        public InputTensor (params int[] shape)
            : base (shape)
        {
        }
    }
}

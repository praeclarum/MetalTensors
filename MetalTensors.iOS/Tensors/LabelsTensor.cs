using System;

namespace MetalTensors.Tensors
{
    public class LabelsTensor : PlaceholderTensor
    {
        public LabelsTensor (params int[] shape)
            : base (shape)
        {
        }
    }
}

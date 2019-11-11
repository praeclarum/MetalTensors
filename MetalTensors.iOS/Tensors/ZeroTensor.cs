using System;

namespace MetalTensors.Tensors
{
    public class ZeroTensor : Tensor
    {
        int[] shape;

        public override int[] Shape => shape;

        public ZeroTensor (int[] shape)
        {
            ValidateShape (shape);
            this.shape = shape;
        }
    }
}

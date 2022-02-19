using System;

namespace MetalTensors.Tensors
{
    public class InputTensor : PlaceholderTensor
    {
        public InputTensor (string? name, params int[] shape)
            : base (name, shape)
        {
        }
        public InputTensor (params int[] shape)
            : this (null, shape)
        {
        }
        protected override TensorHandle CreateHandle (string? label) => new InputHandle (this, label);
    }
}

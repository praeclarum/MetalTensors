using System;

namespace MetalTensors.Tensors
{
    public class InputTensor : PlaceholderTensor
    {
        public InputTensor (string label, params int[] shape)
            : base (label, shape)
        {
        }
        protected override TensorHandle CreateHandle (string? label) => new InputHandle (this, label);
    }
}

using System;

namespace MetalTensors.Tensors
{
    public class LabelsTensor : PlaceholderTensor
    {
        public Tensor OutputTensor { get; }
        public LabelsTensor (string label, Tensor outputTensor, params int[] shape)
            : base (label, shape)
        {
            OutputTensor = outputTensor;
        }
        protected override TensorHandle CreateHandle (string? label) => new LabelsHandle (this, OutputTensor, label);
    }
}

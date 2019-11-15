using System;

namespace MetalTensors.Tensors
{
    public class ModelTensor : Tensor
    {
        public override int[] Shape => Model.Outputs[OutputIndex].Shape;

        public Model Model { get; }
        public int OutputIndex { get; }
        public Tensor[] ModelInputs { get; }

        public ModelTensor (Model model, int outputIndex, params Tensor[] inputs)
        {
            Model = model;
            OutputIndex = outputIndex;
            ModelInputs = inputs;
        }

        public override void Copy (Span<float> destination)
        {
            throw new NotImplementedException ();
        }
    }
}

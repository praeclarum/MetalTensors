using System;
using System.Collections.Generic;
using Metal;
using MetalPerformanceShaders;

namespace MetalTensors.Tensors
{
    public class ModelTensor : Tensor
    {
        public override int[] Shape => BaseModel.Outputs[OutputIndex].Shape;

        public Model BaseModel { get; }
        public int OutputIndex { get; }
        public Tensor[] ModelInputs { get; }

        public override Tensor[] Inputs => ModelInputs;

        public ModelTensor (Model model, int outputIndex, params Tensor[] inputs)
            : base (model.Label)
        {
            BaseModel = model;
            OutputIndex = outputIndex;
            ModelInputs = inputs;
        }

        public override void Copy (Span<float> destination)
        {
            BaseModel.RebuildModelWithInputs (ModelInputs).Outputs[0].Copy (destination);
        }

        public override MPSNNImageNode GetMetalImageNode (MetalImageNodeContext context)
        {
            return BaseModel.RebuildModelWithInputs (ModelInputs).Outputs[0].GetMetalImageNode (context);
        }

        public override Tensor MapInputs (Dictionary<Tensor, Tensor> map)
        {
            return new ModelTensor (BaseModel.MapInputs (map), OutputIndex, ModelInputs.Map (map));
        }
    }
}

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

        public override bool IsStatic => false;

        public ModelTensor (Model model, int outputIndex, params Tensor[] inputs)
            : base (model.Name)
        {
            BaseModel = model;
            OutputIndex = outputIndex;
            ModelInputs = inputs;
        }

        public override Config Config => base.Config.Update (new Config {
            { "model", BaseModel },
            { "outputIndex", OutputIndex },
            { "inputs", ModelInputs },
        });

        public override void Copy (Span<float> destination, IMTLDevice? device = null)
        {
            BaseModel.RebuildModelWithInputs (ModelInputs).Outputs[0].Copy (destination, device);
        }

        public override MPSNNImageNode GetImageNode (MetalImageNodeContext context)
        {
            return BaseModel.RebuildModelWithInputs (ModelInputs).Outputs[0].GetImageNode (context);
        }

        public override Tensor MapInputs (Dictionary<Tensor, Tensor> map)
        {
            return new ModelTensor (BaseModel.MapInputs (map), OutputIndex, ModelInputs.Map (map));
        }

        public override Tensor MapInputs (Func<Tensor, Tensor> map)
        {
            return new ModelTensor (BaseModel.MapInputs (map), OutputIndex, ModelInputs.Map (map));
        }
    }
}

using System;
using System.Collections.Generic;
using Metal;
using MetalPerformanceShaders;

namespace MetalTensors.Tensors
{
    public class LayerTensor : Tensor
    {
        public override int[] Shape => Layer.GetOutputShape (LayerInputs);

        public Layer Layer { get; }
        public Tensor[] LayerInputs { get; }

        public override Tensor[] Inputs => LayerInputs;

        public LayerTensor (Layer layer, Tensor[] inputs)
            : base (layer.Label)
        {
            Layer = layer;
            LayerInputs = inputs;
            Layer.ValidateInputShapes (inputs);
        }

        public override void Copy (Span<float> destination)
        {
            var device = MetalExtensions.Current (null);
            var computed = Layer.ExecuteAsync (LayerInputs, device).Result;
            computed.Copy (destination);
        }

        public override MPSNNImageNode GetMetalImageNode (MetalImageNodeContext context)
        {
            return Layer.GetMetalImageNode (LayerInputs, context);
        }

        public override Tensor MapInputs (Dictionary<Tensor, Tensor> map)
        {
            return new LayerTensor (Layer, LayerInputs.Map (map));
        }

        public override Tensor MapInputs (Func<Tensor, Tensor> map)
        {
            return new LayerTensor (Layer, LayerInputs.Map (map));
        }
    }
}

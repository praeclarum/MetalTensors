using System;
using Metal;
using MetalPerformanceShaders;

namespace MetalTensors.Tensors
{
    public class LayerTensor : Tensor
    {
        public override int[] Shape => Layer.GetOutputShape (LayerInputs);

        public Layer Layer { get; }
        public Tensor[] LayerInputs { get; }

        public LayerTensor (Layer layer, params Tensor[] layerInputs)
        {
            Layer = layer;
            LayerInputs = layerInputs;
        }

        public override void Copy (Span<float> destination)
        {
            var device = MetalExtensions.Current (null);
            var computed = Layer.ExecuteAsync (LayerInputs, device).Result;
            computed.Copy (destination);
        }

        public override MPSNNImageNode GetMetalImageNode (bool training, IMTLDevice device)
        {
            return Layer.GetMetalImageNode (LayerInputs, training, device);
        }
    }
}

using System;

namespace MetalTensors.Tensors
{
    public class LayerOutputTensor : Tensor
    {
        public override int[] Shape => Layer.GetOutputShape (LayerInputs);

        public Layer Layer { get; }
        public Tensor[] LayerInputs { get; }

        public LayerOutputTensor (Layer layer, params Tensor[] layerInputs)
        {
            Layer = layer;
            LayerInputs = layerInputs;
        }

        public override void Copy (Span<float> destination)
        {
            var computed = Layer.OutputAsync (LayerInputs).Result;
            computed.Copy (destination);
        }
    }
}

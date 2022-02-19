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

        public override bool IsStatic => false;

        public LayerTensor (Layer layer, Tensor[] inputs)
            : base (layer.Name)
        {
            Layer = layer;
            LayerInputs = inputs;
            Layer.ValidateInputShapes (inputs);
        }

        public override Config Config => base.Config.Update (new Config {
            { "layer", Layer },
            { "inputs", LayerInputs },
        });

        public override void Copy (Span<float> destination, IMTLDevice? device = null)
        {
            var dev = device.Current ();
            var computed = Layer.ExecuteAsync (LayerInputs, dev).Result;
            computed.Copy (destination, dev);
        }

        public override MPSNNImageNode GetImageNode (MetalImageNodeContext context)
        {
            return Layer.GetImageNode (LayerInputs, context);
        }

        public override Tensor MapInputs (Dictionary<Tensor, Tensor> map)
        {
            return new LayerTensor (Layer, LayerInputs.Map (map));
        }

        public override Tensor MapInputs (Func<Tensor, Tensor> map)
        {
            var newIns = LayerInputs.Map (map);
            if (ReferenceEquals (newIns, LayerInputs))
                return this;
            return new LayerTensor (Layer, newIns);
        }

        public override MPSImage GetMetalImage (IMTLDevice device)
        {
            return ((MPSImageTensor)Layer.ExecuteAsync (LayerInputs, device).Result).MetalImage;
        }
    }
}

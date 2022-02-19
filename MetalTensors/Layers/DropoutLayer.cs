using System;
using Metal;
using MetalPerformanceShaders;

namespace MetalTensors.Layers
{
    public class DropoutLayer : Layer
    {
        public override int MinInputCount => 1;

        public float DropProbability { get; }

        public DropoutLayer (float dropProbability, string? name = null, bool isTrainable = true)
            : base (name, isTrainable: isTrainable)
        {
            DropProbability = dropProbability;
        }

        public override Config Config => base.Config.Add ("dropProbability", DropProbability);

        public override int[] GetOutputShape (params Tensor[] inputs)
        {
            return inputs[0].Shape;
        }

        protected override MPSNNFilterNode CreateFilterNode ((MPSNNImageNode ImageNode, int[] Shape)[] inputs, IMTLDevice device)
        {
            return new MPSCnnDropoutNode (inputs[0].ImageNode, keepProbability: 1.0f - DropProbability);
        }
    }
}

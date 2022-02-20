using System;
using MetalPerformanceShaders;

namespace MetalTensors.Layers
{
    public class ReLULayer : UnopLayer
    {
        public const float DefaultLeakyA = 0.2f;

        public float A { get; }

        public ReLULayer (float a = 0.0f, string? name = null)
            : base (name: name)
        {
            A = a;
        }

        public override Config Config => base.Config.Add ("a", A);

        protected override MPSNNFilterNode CreateUnopNode (MPSNNImageNode imageNode)
        {
            return new MPSCnnNeuronReLUNode (imageNode, A);
        }
    }
}

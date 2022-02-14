using System;
using MetalPerformanceShaders;

namespace MetalTensors.Layers
{
    public class ReLULayer : UnopLayer
    {
        public const float DefaultLeakyA = 0.2f;

        public float A { get; }

        public ReLULayer (float a = 0.0f)
        {
            A = a;
        }

        protected override MPSNNFilterNode CreateUnopNode (MPSNNImageNode imageNode)
        {
            return new MPSCnnNeuronReLUNode (imageNode, A);
        }
    }
}

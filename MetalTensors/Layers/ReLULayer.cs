using System;
using MetalPerformanceShaders;

namespace MetalTensors.Layers
{
    public class ReLULayer : UnopLayer
    {
        public float A { get; }

        public ReLULayer (float a = 0.2f)
        {
            A = a;
        }

        protected override MPSNNFilterNode CreateUnopNode (MPSNNImageNode imageNode)
        {
            return new MPSCnnNeuronReLUNode (imageNode, A);
        }
    }
}

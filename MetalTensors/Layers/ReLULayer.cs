using System;
using MetalPerformanceShaders;

namespace MetalTensors.Layers
{
    public class ReLULayer : UnopLayer
    {
        public float Alpha { get; }

        public ReLULayer (float alpha = 0.2f)
        {
            Alpha = alpha;
        }

        protected override MPSNNFilterNode CreateUnopNode (MPSNNImageNode imageNode)
        {
            return new MPSCnnNeuronReLUNode (imageNode, Alpha);
        }
    }
}

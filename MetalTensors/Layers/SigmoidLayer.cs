using System;
using MetalPerformanceShaders;

namespace MetalTensors.Layers
{
    public class SigmoidLayer : UnopLayer
    {
        public SigmoidLayer (string? name = null)
            : base (name)
        {
        }

        protected override MPSNNFilterNode CreateUnopNode (MPSNNImageNode imageNode)
        {
            return new MPSCnnNeuronSigmoidNode (imageNode);
        }
    }
}

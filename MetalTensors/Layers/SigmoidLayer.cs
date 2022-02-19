using System;
using MetalPerformanceShaders;

namespace MetalTensors.Layers
{
    public class SigmoidLayer : UnopLayer
    {
        public SigmoidLayer (string? name = null, bool isTrainable = true) : base (name, isTrainable: isTrainable)
        {
        }

        protected override MPSNNFilterNode CreateUnopNode (MPSNNImageNode imageNode)
        {
            return new MPSCnnNeuronSigmoidNode (imageNode);
        }
    }
}

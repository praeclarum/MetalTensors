using System;
using MetalPerformanceShaders;

namespace MetalTensors.Layers
{
    public class SigmoidLayer : UnopLayer
    {
        protected override MPSNNFilterNode CreateUnopNode (MPSNNImageNode imageNode)
        {
            return new MPSCnnNeuronSigmoidNode (imageNode);
        }
    }
}

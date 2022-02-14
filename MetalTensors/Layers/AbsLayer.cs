using System;
using MetalPerformanceShaders;

namespace MetalTensors.Layers
{
    public class AbsLayer : UnopLayer
    {
        protected override MPSNNFilterNode CreateUnopNode (MPSNNImageNode imageNode)
        {
            return new MPSCnnNeuronAbsoluteNode (imageNode);
        }
    }
}

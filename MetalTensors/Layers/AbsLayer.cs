using System;
using MetalPerformanceShaders;

namespace MetalTensors.Layers
{
    public class AbsLayer : UnopLayer
    {
        public AbsLayer (string? name = null)
            : base (name)
        {
        }
        protected override MPSNNFilterNode CreateUnopNode (MPSNNImageNode imageNode)
        {
            return new MPSCnnNeuronAbsoluteNode (imageNode);
        }
    }
}

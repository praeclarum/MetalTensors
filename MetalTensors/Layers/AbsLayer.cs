using System;
using MetalPerformanceShaders;

namespace MetalTensors.Layers
{
    public class AbsLayer : UnopLayer
    {
        public AbsLayer (string? name = null, bool isTrainable = true)
            : base (name, isTrainable: isTrainable)
        {
        }
        protected override MPSNNFilterNode CreateUnopNode (MPSNNImageNode imageNode)
        {
            return new MPSCnnNeuronAbsoluteNode (imageNode);
        }
    }
}

using System;
using MetalPerformanceShaders;

namespace MetalTensors.Layers
{
    public class LinearLayer : UnopLayer
    {
        public float Scale { get; }
        public float Offset { get; }

        public LinearLayer (float scale, float offset = 0.0f)
        {
            Scale = scale;
            Offset = offset;
        }

        protected override MPSNNFilterNode CreateUnopNode (MPSNNImageNode imageNode)
        {
            return new MPSCnnNeuronLinearNode (imageNode, Scale, Offset);
        }
    }
}

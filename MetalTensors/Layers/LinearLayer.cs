using System;
using MetalPerformanceShaders;

namespace MetalTensors.Layers
{
    public class LinearLayer : UnopLayer
    {
        public float Scale { get; }
        public float Offset { get; }

        public LinearLayer (float scale, float offset = 0.0f, string? name = null, bool isTrainable = true)
            : base (name, isTrainable: isTrainable)
        {
            Scale = scale;
            Offset = offset;
        }

        public override Config Config => base.Config.Update (new Config {
            { "scale", Scale },
            { "offset", Offset },
        });

        protected override MPSNNFilterNode CreateUnopNode (MPSNNImageNode imageNode)
        {
            return new MPSCnnNeuronLinearNode (imageNode, Scale, Offset);
        }
    }
}

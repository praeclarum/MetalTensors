using System;
using Metal;
using MetalPerformanceShaders;

namespace MetalTensors.Layers
{
    public class DropoutLayer : Layer
    {
        public override int MinInputCount => 1;

        public float KeepProbability { get; }

        public DropoutLayer (float keepProbability)
        {
            KeepProbability = keepProbability;
        }

        public override int[] GetOutputShape (params Tensor[] inputs)
        {
            return inputs[0].Shape;
        }

        protected override MPSNNFilterNode CreateFilterNode ((MPSNNImageNode ImageNode, int[] Shape)[] inputs, IMTLDevice device)
        {
            return new MPSCnnDropoutNode (inputs[0].ImageNode, KeepProbability);
        }
    }
}

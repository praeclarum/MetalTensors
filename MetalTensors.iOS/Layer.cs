using System;
using MetalTensors.Tensors;

namespace MetalTensors
{
    public abstract class Layer
    {
        public abstract int[] GetOutputShape (params Tensor[] inputs);

        public Tensor Output (params Tensor[] inputs)
        {
            return new LayerOutputTensor (this, inputs);
        }

        public abstract Tensor Compute (Tensor[] inputs);
    }
}

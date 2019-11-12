using System;

namespace MetalTensors.Tensors
{
    public class ArrayTensor : Tensor
    {
        readonly float[] data;
        readonly int[] shape;

        public override int[] Shape => shape;

        public ArrayTensor (float[] data)
        {
            this.data = data;
            this.shape = new int[] { data.Length };
        }

        public override void Copy (Span<float> destination)
        {
            ValidateCopyDestination (destination);
            Span<float> dataSpan = data;
            dataSpan.CopyTo (destination);
        }

        public override float this[params int[] indexes] {
            get {
                var i = 0;
                if (indexes.Length > 0)
                    i = indexes[0];
                return data[i];
            }
        }
    }
}
